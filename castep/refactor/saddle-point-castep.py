import os
import sys
import time
import numpy as np
import numpy.linalg as la

# Pad a string s to length l
def pad_str(s, l):
	s = str(s)
	while len(s) < l: s += " "
	while len(s) > l: s = s[:-1]
	return s

class parameter:

	def __init__(self, name, value, scale, fixed=False):

		# Create a parameter
		self.name = name
		self.value = value
		self.scale = scale
		self.fixed = fixed
		self.init_val = value

	def col_vals(self):
		return [str(s) for s in 
		[self.name, self.value,
		 self.init_val, self.scale,
		 self.fixed]]

	@staticmethod
	def col_titles():
		return ["Name","Value","Initially","Scale","Fixed"]


	# Print a nice table of the given parameters  
	@staticmethod
	def table(params, title="Parameters"):
		
		col_titles = parameter.col_titles()
		widths = [len(t) for t in col_titles]
		for p in params:
			for i, c in enumerate(p.col_vals()):
				if len(c) > widths[i]: widths[i] = len(c)
		tot_w = sum(widths) + 2*len(col_titles)
		div = "".join(["%" for i in range(0, tot_w)])
		ret = div + "\n" + title + "\n"
		for i, c in enumerate(col_titles): ret += pad_str(c,widths[i]) + "  "
		ret += "\n" + div + "\n"
		for p in params:
			for i, c in enumerate(p.col_vals()):
				ret += pad_str(c,widths[i]) + "  "
			ret += "\n"
		ret += div

		return ret

class cell:
	def __init__(self, cellfile):

		# Parse a cell from a cellfile
		self.params = []

		# Parse line-by-line, blanking out lines that are parsed
		lines = [l.lower() for l in open(cellfile).read().split("\n")]
		for i, line in enumerate(lines):

			# Parse blocks only
			if not "%block" in line: continue
			lines[i] = ""

			# Parse lattice_abc block
			if "lattice_abc" in line:
				i2 = i+1
				if len(lines[i2].split()) != 3: 
					# Skip unit specification
					lines[i2] = ""
					i2 += 1
			
				# Parse lattice parameters	
				for i, val in enumerate([float(w) for w in lines[i2].split()]):
					pname = "Lattice parameter "+"ABC"[i]
					p = parameter(pname, val, 1.0, False)
					self.params.append(p)

				# Parse lattice angles
				for i, val in enumerate([float(w) for w in lines[i2+1].split()]):
					pname = "Lattice angle "+"ABC"[i]
					p = parameter(pname, val, 90, False)
					self.params.append(p)

				for j in [i2,i2+1,i2+2]: lines[j] = ""

			# Parse positons_frac block
			elif "positions_frac" in line:
				i2 = i+1
				while True:
					if "%endblock" in lines[i2]: 
						lines[i2] = ""
						break
					a,x,y,z = lines[i2].split()
					atom_n = i2-i
					for ic, val in enumerate([float(v) for v in [x,y,z]]):
						pname = "Atom "+str(atom_n)+" ("+a+") "+"xyz"[ic]+" coord"
						p = parameter(pname, val, 1.0, False)
						self.params.append(p)
					lines[i2] = ""
					i2 += 1

		# Extra lines (i.e lines not parsed already)
		self.extras = [l.strip() for l in lines if len(l.strip()) != 0]

	def gen_cellfile(self):

		# Generate a cell file for this cell
		cf  = "%block lattice_abc\n"
		cf += " ".join([str(self.params[i].value) for i in range(0,3)])+"\n"
		cf += " ".join([str(self.params[i].value) for i in range(3,6)])+"\n"
		cf += "%endblock lattice_abc\n\n"
		cf += "%block positions_frac\n"
		for i in range(6,len(self.params),3):
			cf += " ".join([str(self.params[j].value) for j in range(i,i+3)])+"\n"
		cf += "%endblock lattice_abc\n\n"
		for e in self.extras: cf += e + "\n"
		return cf.strip()

	def fix_atoms(self, ns=None, fix=True):
		for i in range(6,len(self.params),3):
			if ns != None and 1+(i-6)/3 not in ns: continue
			for j in range(i,i+3): self.params[j].fixed = fix

	def fix_lattice_params(self, params, fix=True):
		for a in params.lower():
			self.params["abc".index(a)].fixed = fix

	def fix_lattice_angles(self, params, fix=True):
		for a in params.lower():
			self.params["abc".index(a)+3].fixed = fix

	def formatted_cell_file(self, title = "Cell file"):
		cf  = self.gen_cellfile()
		width = max([len(l) for l in cf.split("\n")])
		if len(title) > width: width = len(title)
		div = "".join(["~" for i in range(0,width)])
		return div + "\n" + title + "\n" + div + "\n" + cf + "\n" + div

	def potential(self):
		return 0

def find_saddle_point(cell):

	for step_index in range(0,1):
		
		pot_here = cell.potential()

		forces = {}
		for p in cell.params:
			if p.fixed: continue
			forces[p] = 0

		print forces

c = cell(sys.argv[1])
c.fix_lattice_params("C")
c.fix_lattice_angles("ABC")
print parameter.table(c.params, title = "Initial cell parameters")
print c.formatted_cell_file(title = "Example generated cell file")
find_saddle_point(c)
