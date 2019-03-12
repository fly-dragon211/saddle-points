import os
import numpy as np
import numpy.linalg as la

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

# Run a q.e singlepoint at the given atom positions
singlepoint_count = 1
def run_singlepoint(atom_positions):
	global singlepoint_count

	os.system("mkdir singlepoints 2>/dev/null")
	os.system("cp si_c.in singlepoints")
	set_atom_positions("singlepoints/si_c.in", atom_positions)
	os.system("cd singlepoints; nice -15 mpirun pw.x <si_c.in> si_c.out")
	lines = open("singlepoints/si_c.out").read().split("\n")

	atom_forces = []
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
	
	os.system("mv singlepoints/si_c.out singlepoints/si_c_"+
		   str(singlepoint_count)+".out")
	singlepoint_count += 1

	return [total_energy, atom_forces]

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
	      max_step = 0.1):

	rand_d  = np.random.rand(len(atom_positions))-0.5
	rand_d /= la.norm(rand_d)
	path_p  = [np.array(atom_positions) + max_step*rand_d]

	for iteration in range(0,50):
		
		pot, f  = run_singlepoint(path_p[-1])
		n  = path_p[-1].copy()
		n /= la.norm(n)

		f_para = np.dot(f, n)*n
		f_perp = f - f_para

		f_eff = f_perp - f_para

		if la.norm(f_eff) > max_step:
			f_eff = f_eff * max_step / la.norm(f_eff)

		path_p.append(path_p[-1] + f_eff)
	
		print pot, la.norm(f)


act_relax([0,0,0,1.53104,0.88946356,0.625044463])
