import os
import sys
import time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Pad a string s to length l
def pad_str(s, l):
	s = str(s)
	while len(s) < l: s += " "
	while len(s) > l: s = s[:-1]
	return s

def interp_new(interp_to, xs, fs):

	if len(xs) != len(fs):
		print "Error in interp_new: len(xs) != len(fs)."
		quit()

        M = len(xs)

        m = np.zeros((4*(M-1), 4*(M-1)))
        b = np.zeros(4*(M-1))

        for i in range(0, M-1):

                n  = i*4

                # Left (1) and right(2) boundaries of section
                x1 = xs[i]
                x2 = xs[i+1]
                f1 = fs[i]
                f2 = fs[i+1]

                # Passes through x1, f1 and x2, f2
                b[n] = f1
                b[n+1] = f2
                for p in range(0,4):
                        m[n][n+p] = x1**(p+0.0)
                        m[n+1][n+p] = x2**(p+0.0)

                # Boundary between sections
                n  = i*4
                nl = n-4
                nr = n+4

                if nl >= 0:
                        # Continuous first derivative at left boundary
                        b[n+2] = 0

                        m[n+2][n] = 0
                        m[n+2][n+1] = 1
                        m[n+2][n+2] = 2*x1
                        m[n+2][n+3] = 3*x1**2

                        m[n+2][nl] = 0
                        m[n+2][nl+1] = -1
                        m[n+2][nl+2] = -2*x1
                        m[n+2][nl+3] = -3*x1**2

                        # Continous second derivative at left boundary
                        b[n+3] = 0

                        m[n+3][n] = 0
                        m[n+3][n+1] = 0
                        m[n+3][n+2] = 1
                        m[n+3][n+3] = 3*x1

                        m[n+3][nl] = 0
                        m[n+3][nl+1] = 0
                        m[n+3][nl+2] = -1
			m[n+3][nl+3] = -3*x1

        # Set d^3/dx^3 = 0 at far left/right edge
        # goes in rows 3 and 4 (indicies 2 and 3)
        # as the leftmost section has no left boundary

        b[2] = 0
        m[2][3] = 1
        b[3] = 0
        m[3][(M-2)*4+3] = 1

	minv = la.inv(m)
	coeff = np.dot(minv, b)
        #coeff = np.matmul(minv, b)

	ret = []
	for x in interp_to:
		right_index = None
		for i in range(0, len(xs)):
			if x < xs[i]:
				right_index = i
				break

		if right_index == 0:
			print "Error x requested out of interpolation range! (too small)"
			print x,"<", min(xs)
			quit()
		if right_index == None:
			print "Error x requested out of interpolation range! (too large)"
			print x,">", max(xs)
			quit()

		ci = right_index-1
		c = coeff[ci*4:(ci+1)*4]
		ret.append(c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3)
	return np.array(ret)


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

		# Set the scale of lattice parameters to the average lattice parameter
		av_lat_param = np.mean([abs(p.value) for p in self.params[0:3]])
		for p in self.params[0:3]: p.scale = av_lat_param

		# Extra lines (i.e lines not parsed already)
		self.extras = [l.strip() for l in lines if len(l.strip()) != 0]

	def variable_params(self):
		return [p for p in self.params if not p.fixed]

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
		
		# Fix the atoms labelled by the indicies in ns
		for i in range(6,len(self.params),3):
			if ns != None and 1+(i-6)/3 not in ns: continue
			for j in range(i,i+3): self.params[j].fixed = fix

	def fix_lattice_params(self, params, fix=True):

		# Fix the lattice params specified in the string params
		for a in params.lower():
			self.params["abc".index(a)].fixed = fix

	def fix_lattice_angles(self, params, fix=True):

		# Fix the lattice angles specified in the string params
		for a in params.lower():
			self.params["abc".index(a)+3].fixed = fix

	def formatted_cell_file(self, title = "Cell file"):

		# Returns a nicely formtted cell file for printing to terminal
		cf  = self.gen_cellfile()
		width = max([len(l) for l in cf.split("\n")])
		if len(title) > width: width = len(title)
		div = "".join(["~" for i in range(0,width)])
		return div + "\n" + title + "\n" + div + "\n" + cf + "\n" + div

	def perturb(self, param, delta):
		
		# Perturb the given parameter by the given amount
		if param not in self.params:
			print "Error: Tried to perturb a non-existant cell parameter!"
			return

		param.value += delta

	def reset(self):
		for p in self.params:
			p.value = p.init_val

	def test_pot_coulomb(self, rotation=0):
		disps = []
		for p in self.variable_params():
			disp = (p.value-p.init_val)/p.scale
			disps.append(disp)
		disps = np.array(disps)
		
		if rotation > 0:
			r = la.norm(disps)*rotation
			s = np.sin(r)
			c = np.cos(r)
			m = np.array([[c,s],[-s,c]])
			disps = np.matmul(m, disps)
			
		pot = 0
		xs = [1,1,-1,-1]
		ys = [1,-1,1,-1]
		for x in xs:
			for y in ys:
				pot += 1/la.norm(np.array([x,y])-disps)
		return pot

	def potential(self):
		return self.test_pot_coulomb(rotation=2)

	def line_minimize(self, direction, init_pot=None):
		
		# Minimize the cell potential along the
		# direction given in variable parameter
		# space

		vp = self.variable_params()
		if len(direction) != len(vp):
			print "Error, dimension mismatch in line_minimize!"
			return	

		if init_pot == None: init_pot = self.potential()

		direction /= la.norm(direction)
		init_pars = [p.value for p in vp]

		disps = [0]
		pots  = [init_pot]

		while True:
			
			n = pots.index(min(pots))
			if n not in [0, len(pots)-1]: break

			if n == 0:
				disps.insert(0, disps[0]-0.05)
				for i, v in enumerate(init_pars):
					vp[i].value = v + disps[0]*direction[i]
				pots.insert(0, self.potential())
			else:
				disps.append(disps[-1]+0.05)
				for i, v in enumerate(init_pars):
					vp[i].value = v + disps[-1]*direction[i]
				pots.append(self.potential())

		int_x = np.linspace(min(disps)+0.001, max(disps)-0.001, 100)
		int_y = interp_new(int_x, disps, pots)

		mini = list(int_y).index(min(int_y))
		for i, v in enumerate(init_pars):
			vp[i].value = v + int_x[mini]*direction[i]

		return
		plt.plot(disps, pots)
		plt.plot(int_x, int_y)
		plt.axvline(int_x[mini])
		plt.show()

pot_already_plotted = False
def plot_potential(cell):
	global pot_already_plotted
	if pot_already_plotted: return
	pot_already_plotted = True
	X_RANGE = cell.variable_params()[0].scale*2
	Y_RANGE = cell.variable_params()[1].scale*2
        RESOL = 100
        x   = np.linspace(-X_RANGE,X_RANGE,RESOL)
        y   = np.linspace(-Y_RANGE,Y_RANGE,RESOL)
        z   = []
        all_vals = []
        max_val = -np.inf
        min_val = np.inf
	vp = cell.variable_params()
        for yi in y:
                row = []
                for xi in x:
			vp[0].value = vp[0].init_val + xi
			vp[1].value = vp[1].init_val + yi
			val = cell.potential()
                        row.append(val)
                        all_vals.append(val)
                        if val > max_val : max_val = val
                        if val < min_val : min_val = val
                z.append(row)
	for p in vp: p.value = p.init_val
        z = ((z-min_val)/(max_val-min_val))**(0.25)
        x,y = np.meshgrid(x,y)
        levels = np.linspace(0,1,30)
        plt.contour(x,y,z,cmap="tab20c",levels=levels)
        plt.imshow(z,cmap="tab20c",extent=(-X_RANGE,X_RANGE,-Y_RANGE,Y_RANGE),
                   origin="lower",interpolation="bilinear",alpha=0.2)
        plt.xlabel("x")
        plt.ylabel("y")


class path_info:
	
	def __init__(self):
		self.pot       = None
		self.disp      = None
		self.force     = None
		self.norm      = None
		self.fpara     = None
		self.fperp     = None
		self.dx        = None

def find_saddle_point(cell):

	title = "Begin saddle point search"
	div = "".join(["%" for c in title]) 
	print div + "\n" + title + "\n" + div
	max_par_name_len = max([len(p.name) for p in cell.variable_params()])

	var_param_count = len(cell.variable_params())
	max_ss  = 0.1
	path = []

	# Perturb the cell by an initial random amount
	for i, p in enumerate(cell.variable_params()):
		cell.perturb(p, max_ss*(np.random.rand()*2-1))

	for step_index in range(0,25):
		
		print "\nIteration "+str(step_index+1)

		p       = path_info()
		p.pot   = cell.potential()
		p.force = np.zeros(var_param_count)
		p.disp  = np.zeros(var_param_count)

		print "    Potential  : "+str(p.pot)+" eV"
		tf = "    {0:{1}}   {2:15}   {3:16}"
		header = tf.format("Parameter", max_par_name_len, "Value", "Force")
		tdiv = "    "+"".join(["~" for char in header])[0:-4]
		print "\n    Force evaluation..."
		print tdiv + "\n" + header + "\n" + tdiv

		for i, par in enumerate(cell.variable_params()):
			
			# Caclulate the force on the parameter
			# using finite differences
			eps = max_ss * par.scale / 10.0
			cell.perturb(par, eps)
			grad = cell.potential() - p.pot
			cell.perturb(par, -eps)
			p.force[i] = -grad

			tf = "    {0:{1}} | {2:10.9e} | {3:10.9e}"
			print tf.format(par.name, max_par_name_len, par.value, p.force[i])

			# Record the displacement of this
			# parameter from it's initial value
			p.disp[i] = par.value - par.init_val

		print tdiv
		print "    |Force| = "+str(la.norm(p.force))

		p.norm = p.disp / la.norm(p.disp)

		# Calculate the components of the force which
		# are parrallel and perpendicular to p.norm
		p.fpara = np.dot(p.force, p.norm) * p.norm
		p.fperp = p.force - p.fpara

		# Create a displacement along
		# the perpendicular force, but against the
		# parallel one
		p.dx = p.fperp - p.fpara
		p.dx = max_ss * p.dx / la.norm(p.dx)

		# Take the step
		print "\n    Step evaluation...\n"+tdiv
		print "    {0:{1}}   {2:12}".format("Parameter",max_par_name_len,"Step")
		print tdiv
		for i, par in enumerate(cell.variable_params()):
			cell.perturb(par, p.dx[i])
			print "    {0:{1}} | {2:10.9e}".format(par.name, max_par_name_len, p.dx[i])
		print tdiv

		# Carry out a line minimization along the p.fperp direction
		cell.line_minimize(p.fperp)
		
		# Record information about this step
		path.append(p)

	if var_param_count == 2:

		plt.subplot(1,2,1)
		plot_potential(cell)
		xs = [p.disp[0] for p in path]
		ys = [p.disp[1] for p in path]
		plt.plot(xs,ys,marker="+")
		for p in path: 
			perp_norm = p.fperp/la.norm(p.fperp)
			plt.plot(*zip(*[p.disp, p.disp+perp_norm/10]), color="green")
			plt.plot(*zip(*[p.disp, p.disp+p.norm/10]), color="black")
		
		plt.subplot(4,4,3)
		plt.plot([p.pot for p in path])
		plt.xlabel("Iteration")
		plt.ylabel("Potential")

		plt.subplot(4,4,4)
		plt.plot([la.norm(p.force) for p in path])
		plt.xlabel("Iteration")
		plt.ylabel("|Force|")

		plt.subplot(4,4,7)
		plt.plot([la.norm(path[i].disp-path[i-1].disp) for i in range(1,len(path))])
		plt.xlabel("Iteration")
		plt.ylabel("Step size")

# Run program
cell = cell(sys.argv[1])
cell.fix_lattice_params("C")
cell.fix_lattice_angles("ABC")
cell.fix_atoms()
print parameter.table(cell.params, title = "Initial cell parameters")
print "\n"+cell.formatted_cell_file(title = "Example generated cell file")+"\n"
find_saddle_point(cell)
plt.show()
