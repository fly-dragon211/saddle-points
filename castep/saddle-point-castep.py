import os
import sys
import time
import numpy as np
import numpy.linalg as la

def parse_cell_for_params(seed):
	# Parse the castep .cell file line by line
	# and construct the parameter space
	# (convert everything to lower case)
	global parameters, param_scales, lattice_abc_line, lattice_angles_line
	global atom_counts, atom_lines, init_parameters, init_cell_file_lines
	global param_fixed, request_fixed
	lines = [l.lower() for l in open(seed+".cell").read().split("\n")]
	init_cell_file_lines = list(lines)
	for i, line in enumerate(lines):
		line = line.lower()

		# Line i is the start of a block
		if "%block" in line:

			if "lattice_abc" in line:
				if len(lines[i+1].split()) < 3: i += 1 # Skip units specification

				# Read in lattice parameters as parameters
				cell_params = [float(e) for e in lines[i+1].split()]
				lattice_abc_line = i+1
				parameters["lattice_a"] = cell_params[0]
				parameters["lattice_b"] = cell_params[1]
				parameters["lattice_c"] = cell_params[2]
				param_scales["lattice_a"] = 1.0
				param_scales["lattice_b"] = 1.0
				param_scales["lattice_c"] = 1.0

				# Read in lattice angles as parameters
				angles = [float(e) for e in lines[i+2].split()]
				lattice_angles_line = i+2
				parameters["angle_a"] = angles[0]
				parameters["angle_b"] = angles[1]
				parameters["angle_c"] = angles[2]
				param_scales["angle_a"] = 180/np.pi
				param_scales["angle_b"] = 180/np.pi
				param_scales["angle_c"] = 180/np.pi

			if "positions_frac" in line:
				i2 = i+1
				while True:
					# Read in atom fractional coords as parameters
					if "%endblock" in lines[i2]: break
					a,x,y,z = lines[i2].split()	
					if a in atom_counts: atom_counts[a] += 1
					else: atom_counts[a] = 1
					name = a+"_atom_"+str(atom_counts[a])
					atom_lines[i2] = name
					parameters[name+"_x"] = float(x)
					parameters[name+"_y"] = float(y)
					parameters[name+"_z"] = float(z)
					param_scales[name+"_x"] = 1.0
					param_scales[name+"_y"] = 1.0
					param_scales[name+"_z"] = 1.0
					if i2 == i+1:
						# First atom is fixed
						param_fixed[name+"_x"] = True
						param_fixed[name+"_y"] = True
						param_fixed[name+"_z"] = True
					i2 += 1

	for p in parameters:
		if p in param_fixed: continue
		param_fixed[p] = False
	init_parameters = parameters.copy()

def pad_str(s, l):
	for i in range(0, l-len(s)): s += " "
	while len(s) > l: s = s[:-1]
	return s

def print_params():
	# Output parameter configuration
	global parameters, init_parameters
	max_len = max([len(str(p)) for p in parameters])
	
	div = ""
	for i in range(0, (max_len+1)*5): div += "%"

	print div
	print "Total parameters : " + str(len(parameters))

	padded      = pad_str("Parameter", max_len)
	padded_val  = pad_str("Value", max_len)
	padded_init = pad_str("Initially", max_len)
	padded_fix  = pad_str("Fixed", max_len)

	print padded + " " + padded_val + " " + padded_init + " " + padded_fix + " Scale" 
	print div

	for p in parameters:
		padded          = pad_str(str(p), max_len)
		padded_val      = pad_str(str(parameters[p]), max_len)
		padded_init_val = pad_str(str(init_parameters[p]), max_len)
		padded_fixed    = pad_str(str(param_fixed[p]), max_len)
		padded_scale    = pad_str(str(param_scales[p]), max_len)
		print padded+" "+padded_val+" "+padded_init_val+" "+padded_fixed+" "+padded_scale

	print div

def params_to_cell():
	# Convert the current parameter set back into a cell file
	global init_cell_file_lines, lattice_abc_line, lattice_angles_line
	ret = ""
	for i in range(0,len(init_cell_file_lines)):
		if i == lattice_abc_line:
			# Replace lattice abc with new vals
			ret += str(parameters["lattice_a"]) + " "
			ret += str(parameters["lattice_b"]) + " "
			ret += str(parameters["lattice_c"]) + "\n"

		elif i == lattice_angles_line:
			# Replace lattice angles with new vals
			ret += str(parameters["angle_a"]) + " "
			ret += str(parameters["angle_b"]) + " "
			ret += str(parameters["angle_c"]) + "\n"
		elif i in atom_lines:
			# Replace atom frac coords with new vals
			name = atom_lines[i]
			symbol = name.split("_")[0]
			symbol = symbol[0].upper() + symbol[1:]
			ret += symbol + " "
			ret += str(parameters[name+"_x"]) + " "
			ret += str(parameters[name+"_y"]) + " "
			ret += str(parameters[name+"_z"]) + "\n"
		else:
			# Otherwise maintain initial line
			ret += init_cell_file_lines[i] + "\n"
	return ret.strip()

def print_cell():
	# Print the cell file with the current params to screen
	c = params_to_cell()
	max_len = max([len(l) for l in c.split("\n")])
	div = ""
	for i in range(0, max_len): div += "~"
	print div
	print "Cell file from params"
	print div
	print c
	print div

def potential():
	# Calculate the potential at the given parameter set
	global pot_seed_number, castep_cmd
	pot_seed_name = sys.argv[1] + "_" + str(pot_seed_number)
	pot_seed_number += 1

	# Copy the .param file
	os.system("cp "+sys.argv[1]+".param "+"./singlepoints/"+pot_seed_name+".param")

	# Create the .cell file
	cell_file = open("./singlepoints/"+pot_seed_name+".cell", "w")
	cell_file.write(params_to_cell())
	cell_file.close()
	
	# Run castep
	os.system("cd ./singlepoints; "+castep_cmd+" "+pot_seed_name+" >/dev/null 2>&1")
	
	# Extract results
	lines = [l.lower() for l in open("./singlepoints/"+pot_seed_name+".castep").read().split("\n")]
	for i, line in enumerate(lines):
		if "nb est. 0k energy" in line:
			return float(line.split("=")[1].split("e")[0])

def pot(x,names):
	# Overload of the above when x is an np.array which
	# stores the parameter)named "names[i]" at x[i]
	for i, n in enumerate(names): parameters[n] = x[i]
	return potential()

def init_files():
	# Empty the ./singlepoints directory
	if os.path.exists("./singlepoints"):
		for f in os.listdir("./singlepoints"):
			os.remove("./singlepoints/"+f)
		os.rmdir("./singlepoints")
	os.makedirs("./singlepoints")

def act_relax():
	global parameters, param_scales, param_fixed, initial_time

	step_size = 0.1 * np.sqrt(float(len(parameters)))

	div = ""
	for n in range(0,50): div += "="
	print div
	print "Begin activation-relaxation"
	print div

	# Construct a map from a parameter
	# index to it's name
	names  = []
	scales = []
	for p in parameters:
		names.append(p)
		scales.append(param_scales[p])
	
	# Output that map
	print "Parameter indexing: "
	max_par_name_len = max([len(p) for p in parameters])
	for i, n in enumerate(names):
		print pad_str(n, max_par_name_len), "->",pad_str(str(i),4),":",parameters[names[i]]
	print div

	path = []
	path_pot = []
	path_force = []

	# Initial position
	x = np.array([parameters[n] for n in names])

	for iter_index in range(0,20):

		print "Iteration ", (iter_index+1), " time so far: ", time.time()-initial_time, "s"
		
		# Evaluate the potential at current location
		pot_here = pot(x, names)
		print "    Potential : " + str(pot_here) + " eV"

		# Evaluate the gradient at current location
		grad = np.zeros(len(x))
		print "    Force components"
		for i in range(0,len(x)):
			append = ""
			if param_fixed[names[i]]:
				grad[i] = 0
				append = " (fixed)"
			else:
				eps = step_size / 10
				di = np.zeros(len(x))
				di[i] = eps * scales[i]
				grad[i] = pot(x + di, names) - pot_here
				grad[i] /= eps
			print "        "+names[i]+" ( = "+str(x[i])+") : "+ str(-grad[i]) + append

		if iter_index == 0:
			# For first iteration, take a
			# step from initial position
			print "    Random init step ..."
			dx = np.zeros(len(x))
			for i in range(0,len(x)):
				if not param_fixed[names[i]]:
					dx[i] = step_size*scales[i]*(np.random.rand()*2-1)
		else:
			# Invert radial component
			# of force and set as direction to move
			f = -grad
			n = x/la.norm(x)
			fpara = np.dot(f,n)*n
			fperp = f - fpara
			dx = fperp - fpara

		# Once we are near the saddle point, bisect to 
		# get closer to it by reducing the step size
		if False:
			if np.dot(path[-1]-path[-2], dx) < 0:
				print "    Bisecting step ..."
				step_size /= 2

		dx = step_size * dx / la.norm(dx)
		print "    Step to take (step size = "+str(la.norm(dx))+")"
		for i in range(0,len(dx)):
			print "        "+names[i]+" : "+str(dx[i])

		# Make move
		x += dx

		# Record values
		path.append(x.copy())
		path_force.append(-grad)
		path_pot.append(pot_here)

	# Write path information to file
	path_info = ""
	for i in range(0, len(path)):
		for xi in path[i]: path_info += str(xi) + " "
		path_info += ":" + str(path_pot[i]) + ":"
		for fi in path_force[i]: path_info += str(fi) + " "
		path_info += "\n"
	open("./path_info", "w").write(path_info)

# Global variables
parameters = {}
param_fixed = {}
param_scales = {}
init_parameters = {}
init_cell_file_lines = []
atom_counts = {}
atom_lines = {}
lattice_abc_line = -1 
lattice_angles_line = -1
pot_seed_number = 1
castep_cmd = "mpirun castep.mpi"
if "serial" in sys.argv[1:]: castep_cmd = "castep.serial"
request_fixed = []
for a in sys.argv[1:]: 
	if a.startswith("fix_"): 
		request_fixed.append(a[len("fix_"):])
print request_fixed
initial_time = time.time()

init_files()
parse_cell_for_params(sys.argv[1])
print_params()
print_cell()
act_relax()
