import numpy as np
import os

# The cell object represents the current
# simulation cell as a set of atoms and
# fractional coordinates and a set of
# lattice vectors
class cell(object):
	
	# Constructor which parses a q.e file
	def __init__(self, template):

		with open(template) as f:
			lines = f.read().split("\n")

		for i, line in enumerate(lines):

			# Parse the number of atoms
			if "nat" in line:
				atom_count = int(line.split("=")[1].replace(",",""))
			
			# Parse the lattice from a CELL_PARAMETERS block
			if "CELL_PARAMETERS" in line:
				if "angstrom" not in line:
					raise Exception("only CELL_PARAMETERS in "+
							"angstrom is supported!")
				self.lattice = []
				for j in range(i+1, i+4):
					row = [float(w) for w in lines[j].split()]
					self.lattice.append(row)


			# Parse the atoms from an ATOMIC_POSITIONS block
			if "ATOMIC_POSITIONS" in line:
				if "crystal" not in line:
					raise Exception("Only ATOMIC_POSITIONS in "+
							"crystal coords is supported!")
				self.atoms = []
				for j in range(i+1, i+1+atom_count):
					spl = lines[j].split()
					a   = [spl[0]]
					a.extend([float(w) for w in spl[1:4]])
					self.atoms.append(a)

		self.template = template
		self.lattice  = np.array(self.lattice)

	# Return a copy of this cell
	def copy(self):
		ret = cell(self.template)
		ret.atoms       = list(self.atoms)
		ret.lattice     = self.lattice.copy()
		ret.stress_cart = self.stress_cart.copy()
		ret.forces_cart = self.forces_cart.copy()
		return ret

	# Get a string containing info about this cell
	def info(self):
		inf  = "Lattice\n"
		for row in self.lattice: 
			inf += "{0:15.10g}  {1:15.10g}  {2:15.10g}\n".format(*row)

		inf += "Atoms (fractional coords)\n"
		for a in self.atoms: 
			inf += "{0:5.5}  {1:10.5g}  {2:10.5g}  {3:10.5g}\n".format(*a)
		
		return inf.strip()
	
	# Get the atoms in cartesian coordinates
	def atoms_cart(self, include_name=True):
		ret = []
		for a in self.atoms:
			pos = np.dot(np.array(a[1:4]).T, self.lattice)
			if include_name:
				ret.append([a[0], pos[0], pos[1], pos[2]])
			else:
				ret.append([pos[0], pos[1], pos[2]])
		if include_name:
			return ret
		else:
			return np.array(ret)

	# Perturb the atoms in cartesian coordinates
	def perturb_atoms_cart(self, pertubations):
		lat_tr_inv = np.linalg.inv(self.lattice.T)
		for i, a in enumerate(self.atoms):
			pos_cart  = np.dot(np.array(a[1:4]).T, self.lattice)
			pos_cart += np.array(pertubations[i])
			pos_frac  = np.dot(lat_tr_inv, pos_cart.T)
			for j in range(0,3):
				self.atoms[i][1+j] = pos_frac[j]

	# Apply a strain tensor (distortion of space) according
	# to r_i -> (delta_ij + strain_ij)*r_j for all positions r
	# this is achived by simply applying the strain to the
	# lattice vectors, leaving fractional coordinates untouched
	def apply_strain_cart(self, strain):
		self.lattice = np.dot(np.identity(3) + strain, self.lattice)

	# Run a singlepoint calculation at this cell config
	# takes a template file with the settings etc and simply
	# replaces the cell coordinates etc
	def run_singlepoint_qe(self, template_file):

		# Create the singlepoint directory if it doesn't exist
		if not os.path.isdir("singlepoints/"):
			os.mkdir("singlepoints/")

		# Count singlepoints so that we can
		# label them sequentially
		spc = 0
		for f in os.listdir("singlepoints"):
			if f.endswith(".in"):
				spc += 1

		# Read the template file in by lines
		with open(template_file) as f:
			lines = f.read().split("\n")

		# Create the q.e input file for this singlepoint run
		infile  = "sp_"+str(spc)+".in"
		outfile = "sp_"+str(spc)+".out" 
		input_file = open("singlepoints/"+infile, "w")

		iignored = []
		for i, line in enumerate(lines):

			# Ignore particular lines
			if i in iignored: continue

			# Overwrite CELL_PARAMETERS section
			if "CELL_PARAMETERS" in line:
				iignored.extend([i+1,i+2,i+3])
				input_file.write("CELL_PARAMETERS (angstrom)\n")
				for j in range(0, 3):
					input_file.write(" ".join(
					str(a) for a in self.lattice[j])+"\n")
				continue

			# Overwrite ATOMIC_POSITIONS section
			if "ATOMIC_POSITIONS" in line:
				for j in range(0,len(self.atoms)):
					iignored.append(i+j+1)
				input_file.write("ATOMIC_POSITIONS (crystal)\n")
				for a in self.atoms:
					input_file.write(" ".join(
					str(ai) for ai in a)+"\n")
				continue

			# Just copy this line from the template file
			input_file.write(line+"\n")

		input_file.close()

		# Run pw.x
		os.system("cd singlepoints; mpirun pw.x <"+infile+"> "+outfile)

		# Read output file lines
		with open("singlepoints/"+outfile) as f:
			lines = f.read().split("\n")

		# Parse output file
		self.stress_cart = []
		self.forces_cart = []
		for i, line in enumerate(lines):

			# Parse stress tensor (in GPa)
			if "P=" in line:
				for j in range(1,4):
					row = [float(w)/10 for w in lines[i+j].split()[3:6]]
					self.stress_cart.append(row)

			# Parse forces on atoms (in Ry/au)
			if "Forces acting on atoms" in line:
				j = i
				while True:
					j += 1	
					if "Total force" in lines[j]: break
					spl = lines[j].split("=")
					if len(spl) < 2: continue
					spl = spl[-1].split()
					self.forces_cart.append([float(w) for w in spl])

		self.stress_cart = np.array(self.stress_cart)
		self.forces_cart = np.array(self.forces_cart)

	@property
	def total_force(self):
		return np.linalg.norm([x for vec in self.forces_cart for x in vec])

	@property
	def total_stress(self):
		return np.linalg.norm([x for row in self.stress_cart for x in row])

	@property
	def pressure(self):
		return sum(self.stress_cart[i][i] for i in range(0,3))/3
