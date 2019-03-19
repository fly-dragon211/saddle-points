import os
import numpy as np
import numpy.linalg as la
import subprocess as sub

# Set the atom positions in a q.e input file
def set_atom_positions(in_file, atom_positions):
	
	f = open(in_file)
	lines = f.read().split("\n")
	f.close()

	ap3s = []
	for i in range(0,len(atom_positions)/3):
		ap = []
		for j in range(0,3):
			ap.append(atom_positions[i*3+j])
		ap3s.append(ap)

	f = open(in_file, "w")
	i_written_to = 0
	for i in range(0, len(lines)):
		f.write(lines[i]+"\n")
		if "ATOMIC_POSITIONS" in lines[i]:
			for di, p in enumerate(ap3s):
				ap = lines[i+1+di].split()[0]
				ap += " "+str(p[0])
				ap += " "+str(p[1])
				ap += " "+str(p[2])
				f.write(ap+"\n")
			i_written_to = i + 1 + len(ap3s)
			break

	for i in range(i_written_to, len(lines)):
		f.write(lines[i]+"\n")

# Set the cell parametres in a q.e input file
def set_cell_parameters(in_file, cell_params):
	
	f = open(in_file)
	lines = f.read().split("\n")
	f.close()

	f = open(in_file, "w")

	i_cell = -100000
	for i, l in enumerate(lines):
		if "CELL_PARAMETERS" in l: i_cell = i + 1

		if i - i_cell in [0, 1, 2]:
			f.write(" ".join([str(p) for p in cell_params[i-i_cell]])+"\n")
		else:
			f.write(l+"\n")	
		

# Postprocess after a singlepoint calculation
def postprocess(infile):
	f = infile
	lines = open(f).read().split("\n")
        tmp = open("tmp.in","w")
        for l in lines:
                if "pseudo" in l: continue
                tmp.write(l+"\n")

        tmp.close()
        os.system("c2x --cell_abc tmp.in "+f+".cell 2>&1 1>/dev/null")
        os.system("c2x -e=0.01 "+f+".cell --int 2>"+f+".symm")

	script  = 'load "'+f.split("/")[-1]+'.cell" {2 2 2}\n'
	script += "center *\n"
	#script += "set perspectiveDepth on\n"
	#script += "set cameraDepth 0.5\n"
	script += 'write image 500 500 PNG -1 "'+f.split("/")[-1]+'.png"\n'

	fj = open("singlepoints/jmol_script","w")
	fj.write(script)
	fj.close()

	with open(os.devnull, "w") as devnull:
		sub.Popen("jmol --silent -n -s jmol_script",
			   cwd="singlepoints", shell=True,
			   stdout = devnull, stderr = devnull)

# Run a q.e singlepoint at the given atom position and cell parameters
singlepoint_count = 1
def run_singlepoint(atom_positions, cell_parameters):
	global singlepoint_count

	os.system("mkdir singlepoints 2>/dev/null")
	os.system("cp si_c.in singlepoints")
	set_atom_positions("singlepoints/si_c.in", atom_positions)
	set_cell_parameters("singlepoints/si_c.in", cell_parameters)
	os.system("cd singlepoints; nice -15 mpirun pw.x -nk 4 <si_c.in> si_c.out")
	lines = open("singlepoints/si_c.out").read().split("\n")

	atom_forces = []
	stress_tensor = []
	total_energy = None

	for i, l in enumerate(lines):
		if "Forces acting on atoms" in l:
			j = i
			while True:
				j += 1
				if "SCF" in lines[j]: break
				spl = lines[j].split()
				try:
					f = [float(w) for w in [spl[-3], spl[-2], spl[-1]]]
					atom_forces.extend(f)
				except:
					continue
		if "!" in l:
			total_energy = float(l.split("=")[-1].split("R")[0])

		if "(Ry/bohr**3)" in l and "stress" in l:
			for j in range(i+1, i+4):
				stress_tensor.append([float(w) for w in lines[j].split()[0:3]])
	
	os.system("mv singlepoints/si_c.out singlepoints/si_c_"+
		   str(singlepoint_count)+".out")
	os.system("mv singlepoints/si_c.in  singlepoints/si_c_"+
		   str(singlepoint_count)+".in")
	postprocess("singlepoints/si_c_"+
		   str(singlepoint_count)+".in")
	
	singlepoint_count += 1

	return [total_energy, atom_forces, stress_tensor]

# Apply newton-raphson to the atomic forces
# to find a local stationary point on the B.O.S
def newton_raphson(atom_positions):
	path_p = [np.array(atom_positions)]
	path_f = [get_force(path_p[0])]
	path_p.append(path_p[0] + 0.1*np.random.rand(len(path_p[0])))
	
	for min_iter in range(0, 50):

		pot, f = run_singlepoint(path_p[-1])
		new_pos = np.zeros(len(path_p[-1]))

		print path_p[-1], la.norm(f)

		for i in range(0,len(path_p[-1])):
			denom = path_f[-1][i] - f[i]
			if denom == 0: ratio = 1
			else: ratio = f[i]/denom
			new_pos[i] = path_p[-1][i] + ratio * (path_p[-1][i] - path_p[-2][i])

		path_f.append(f)
		path_p.append(new_pos)

# Carry out activation-relaxation to find a saddle point
def act_relax(atom_positions,
	      cell_parameters,
	      max_step = 0.5):

	start_pos  = atom_positions
	start_cell = cell_parameters
	start_pot, start_f, start_stress = run_singlepoint(start_pos, start_cell)

	rand_d  = np.random.rand(len(atom_positions))-0.5
	for i in range(0,3): rand_d[i] = 0
	rand_d /= la.norm(rand_d)
	path_p  = [np.array(atom_positions) + max_step*rand_d]
	path_c  = [start_cell]

	for iteration in range(0,50):
		
		pot, f, stress = run_singlepoint(path_p[-1], path_c[-1])
		n  = path_p[-1].copy() - start_pos
		n /= la.norm(n)

		f_para = np.dot(f, n)*n
		f_perp = f - f_para

		f_eff = f_perp - f_para

		for i in range(0,3): f_eff[i] = 0
		if la.norm(f_eff) > max_step: f_eff = f_eff * max_step / la.norm(f_eff)

		path_p.append(path_p[-1] + f_eff)
		path_c.append(path_c[-1])

		path_c[-1] = np.matmul(np.identity(3) +  stress, path_c[-1])
	
		print pot, la.norm(f), la.norm(path_p[-1]-path_p[-2])

		if la.norm(f) < 10e-5:
			print "Saddle point reached!"
			break

os.system("rm -r singlepoints")
atom_pos = [0,0,0,1.53104,0.88946356,0.625044463]
cell_par = [[3.088679993,  0.000000000,  0.000000000],
	    [1.544339996,  2.674875369,  0.000000000],
	    [1.544339996,  0.891625156,  2.526982335]]
act_relax(atom_pos, cell_par)
