import os
import sys
import time
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

def interpolate(interp_to, xs, fs):

	# Perform a 1D minimum-curvature interpolation 
	# passing through the points (xs, fs). Evaluates
	# the interpolation at the x values in interp_to
	if len(xs) != len(fs):
		print "Error in interpolation: len(xs) != len(fs)."
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


class parameter(object):

	def __init__(self, name, value, scale, fixed=False):

		# Create a parameter
		self.name = name
		self.value = value
		self.scale = scale
		self.fixed = fixed
		self.init_val = value

class cell(object):
	def __init__(self, cellfile):

		self.params = []
		self.atom_names = []
		self.pot_evals = 0
		self.seed = cellfile.split("/")[-1].split(".cell")[0]
		self.singlepoint_count = 0
		self.test_potential = False

		# Parse a cell from a cellfile
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
						aname = a[0].upper() + a[1:]
						pname = "Atom "+str(atom_n)+" ("+aname+") "+"xyz"[ic]+" coord"
						p = parameter(pname, val, 1.0, False)
						self.params.append(p)
						self.atom_names.append(aname)
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
			cf += self.atom_names[i-6] + " "
			cf += " ".join([str(self.params[j].value) for j in range(i,i+3)])+"\n"
		cf += "%endblock positions_frac\n\n"
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

	def unfix_all(self):

		# Unfix all parameters
		for p in self.params:
			p.fixed = False

	def test_pot(self, rotation=0):

		# A test potential consisting of 4 coulomb
		# centres with an optional rotation
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

		x = disps[0]
		y = disps[1]
		return -np.cos(x*np.pi)*np.cos(y*np.pi)
			
		pot = 0
		xs = [1,1,-1,-1, 2, -2]
		ys = [1,-1,1,-1, 2, -2]
		for x in xs:
			for y in ys:
				norm = la.norm(np.array([x,y])-disps)
				if norm != 0: pot += 1/norm
				else: pot += 100
		return pot

	def test_force(self, rotation=0, fd_eps=0.01):
		
		# Get a test force from the test potential
		# using finite differences
		vp  = self.variable_params()
		ret = np.zeros(len(vp))
		c0  = self.config
		p0  = self.test_pot(rotation=rotation)
		for i, p in enumerate(vp):
			ei = np.zeros(len(vp))	
			ei[i] = fd_eps
			self.config += ei
			ret[i] = -(self.test_pot(rotation=rotation)-p0)/fd_eps
			self.config -= ei
		return ret

	def potential_and_force(self, fd_eps=0.01):

		self.singlepoint_count += 1

		if fd_eps < 10e-4: fd_eps = 10e-4
		if self.test_potential:
			# Get the force and potential from a test potential
			rot = 1
			return [self.test_pot(rotation=rot),
				self.test_force(fd_eps=fd_eps, rotation=rot)]

		# Get the force and potential from castep
		# Run castep with the current configuration
		global castep_cmd
		prefix = "singlepoints/"+self.seed+"_"+str(self.singlepoint_count)
		cellf = open(prefix+".cell","w+")
		paraf = open(prefix+".param","w+")
		cellf.write(self.gen_cellfile())
		paraf.write(open(sys.argv[1]+".param").read())
		cellf.close()
		paraf.close()
		os.system("cd singlepoints; "+castep_cmd+" "+
			  self.seed+"_"+str(self.singlepoint_count)+
			  " >/dev/null 2>&1")

		pot = None
		par_force = {}

		# Parse lattice (used to convert forces
		# from cartesian into fractional)
		lattice = None
		lines = open(prefix+".castep").read().split("\n")
		for i, line in enumerate(lines):

			# Parse lattice vectors
			if "Real Lattice(A)" in line:
				lattice = np.zeros([3,3])
				for j in range(0,3):
					lattice[j] = [float(w) for w in lines[i+1+j].split()[0:3]]

		if lattice is None:
			print "Error: could not read real lattice from castep file: "+prefix+".castep!"
			quit()

		# lti = (lattice^T) ^ -1
		lti = la.inv(lattice.T)

		# Get potential and forces on atoms from castep
		for i, line in enumerate(lines):

			# Parse forces on atoms
			if "Cartesian components (eV/A)" in line:

				j = i
				while True:
					j += 1
					if "." in lines[j]:
						s1, name, num, x, y, z, s2 = lines[j].split()
						for k in range(0, len(self.atom_names)/3):
							if self.atom_names[k*3] != name: continue
							if self.params[6+k*3] in par_force: continue
							xyz = np.array([float(w) for w in [x,y,z]])
							frac = np.dot(lti, xyz)
							par_force[self.params[6+k*3]]   = frac[0]
							par_force[self.params[6+k*3+1]] = frac[1]
							par_force[self.params[6+k*3+2]] = frac[2]

					if "**" in lines[j]:
						break

			# Parse potential
			if "NB" in line:
				pot = float(line.split("=")[1].split("e")[0])

		
		# Get forces corresponding to variable parameters
		vp = self.variable_params()
		force = np.array([par_force[p] for p in vp])
		return [pot, force]
		
		# Get the force using finite differences
		if start_pot == None: start_pot = self.potential()
		start_config = self.config

		force = np.zeros(len(self.config))
		for i in range(0, len(self.config)):
			ei    = np.zeros(len(self.config))
			ei[i] = fd_eps
			self.config = start_config + ei
			force[i]  = -(self.potential() - start_pot)/fd_eps

		self.config = start_config
		return [start_pot, force]

	def potential(self):
		return self.potential_and_force()[0]

	@property
	def config(self):

		# Get the current location in normalized configuration space
		return np.array([p.value/p.scale for p in self.variable_params()])

	@config.setter
	def config(self, cfg):
		
		# Sets the values of the variable parameters
		# from normalized configuration space
		for i, par in enumerate(self.variable_params()):
			par.value = cfg[i]*par.scale

	@property
	def init_config(self):

		# Get the initial location in normalized configuration space
		return np.array([p.init_val/p.scale for p in self.variable_params()])

	def reset(self):

		# Reset the cell to it's initial configuration
		self.config = self.init_config
		self.singlepoint_count = 0

	def line_min_config(self, step, config_dir, max_delta):

		if la.norm(config_dir) == 0: return self.potential()

		# Perform a line minimization of the potential
		# along the direction in configuration space config_dir
		# maximum allowed movement along config_dir is max_delta
		config_dir /= la.norm(config_dir)
		init_config = self.config
		deltas = [0]
		pots   = [self.potential()]

		# Bound the minimum
		while True:
			n = pots.index(min(pots))
			if n not in [0, len(pots)-1]: break

			if n == 0:
				# Minimum is at (deltas[0], pots[0])
				if abs(deltas[0]-step) > max_delta: break
				deltas.insert(0, deltas[0]-step)
				self.config = init_config+deltas[0]*config_dir
				pots.insert(0, self.potential())
			else:
				# Minimum is at (deltas[-1], pots[-1])
				if abs(deltas[-1]+step) > max_delta: break
				deltas.append(deltas[-1]+step)
				self.config = init_config+deltas[-1]*config_dir
				pots.append(self.potential())

		# Use a minimum-curvature interpolation to 
		# approximate the minimal configuration
		int_x = np.linspace(min(deltas)+step/100, max(deltas)-step/100, 100)
		int_y = interpolate(int_x, deltas, pots)
		mini = list(int_y).index(min(int_y))
		self.config = init_config + int_x[mini]*config_dir

		return int_y[mini]

		# Plot the interpolation
		plt.plot(deltas, pots)
		plt.plot(int_x, int_y)
		plt.axvline(int_x[mini])
		plt.show()


	def plot_potential(self):
		
		# Plot the 2D slice of the cell potential
		# optained by varying the first two parameters
		vp   = self.variable_params()
		x    = np.linspace(-2, 2, 50) + self.init_config[0]
		y    = np.linspace(-2, 2, 50) + self.init_config[1]
		z    = []
		extent = (min(x)*vp[0].scale, max(x)*vp[0].scale,
			  min(y)*vp[1].scale, max(y)*vp[1].scale)
		max_val  = -np.inf
		min_val  = np.inf
		for yi in y:
			row = []
			for xi in x:
				self.config = [xi,yi]
				val = self.potential()
				row.append(val)
				if val > max_val : max_val = val
				if val < min_val : min_val = val
			z.append(row)

		z = ((z-min_val)/(max_val-min_val))**(0.25)
		x,y = np.meshgrid(x*vp[0].scale,y*vp[1].scale)
		levels = np.linspace(0,1,50)
		plt.contour(x,y,z,cmap="tab20c",levels=levels)
		plt.imshow(z,cmap="tab20c",extent=extent,
			   origin="lower",interpolation="bilinear",alpha=0.2,aspect="auto")
		plt.xlabel(vp[0].name)
		plt.ylabel(vp[1].name)
		self.reset()

class path_info(object):

	# A class for recording information about the path
	# taken through configuration space
	
	def __init__(self):
		self.pot         = None
		self.norm        = None
		self.force       = None
		self.fpara       = None
		self.fperp       = None
		self.config      = None
		self.path_name   = None
		self.relaxation  = None
		self.activation  = None

	def verbose_info(self, cell):
		vp = cell.variable_params()
		ra = range(0,len(vp))
		s  = ""
		s += "\nPath:"+str(self.path_name)
		s += "\nPotential:"+str(self.pot)
		s += "\nName:"+",".join([p.name for p in vp])
		s += "\nValue:"+",".join([str(self.config[i]*vp[i].scale) for i in ra])
		s += "\nNormal:"+",".join([str(self.norm[i]*vp[i].scale) for i in ra])
		s += "\nForce:"+",".join([str(self.force[i]) for i in ra])
		s += "\nForce parallel:"+",".join([str(self.fpara[i]) for i in ra])
		s += "\nForce perpendicular:"+",".join([str(self.fperp[i]) for i in ra])
		s += "\nActivation:"+",".join([str(self.activation[i]*vp[i].scale) for i in ra])
		s += "\nRelaxation:"+",".join([str(self.relaxation[i]*vp[i].scale) for i in ra])
		return s		

	def force_info(self, cell):
		s  = "Potential : "+str(self.pot)+"\n"
		fs = "{0:20.20}  {1:10.10}  {2:10.10}  {3:10.10}"
		h  = fs.format("Parameter","Value","Initially","Force")
		dv = "".join(["~" for c in h])
		s += dv + "\n" + h + "\n" + dv
		vp  = cell.variable_params()
		for i in range(0,len(vp)):
			par = vp[i].name
			val = self.config[i]*vp[i].scale
			ini = vp[i].init_val
			frc = self.force[i]
			fs  = "{0:20.20}  {1:10.5g}  {2:10.5g}  {3:10.5g}"
			s  += "\n"+fs.format(par,val,ini,frc)
		return s

	def step_info(self, cell, step_size):
		s  = "Normalized step size: "+str(step_size)
		fs = "{0:20.20}  {1:10.10}  {2:10.10}  {3:10.10}"
		h  = fs.format("Parameter","Activation","Relaxation","Step")
		dv = "".join(["~" for c in h])
		s += "\n" + dv + "\n" + h + "\n" + dv
		vp = cell.variable_params()
		for i in range(0,len(vp)):
			par = vp[i].name
			act = self.activation[i]*vp[i].scale
			lim = self.relaxation[i]*vp[i].scale
		        s += "\n{0:20.20}  ".format(par)
			s +="{0:10.5g}  {1:10.5g}  {2:10.5g}".format(act,lim,act+lim)
		return s

	@staticmethod
	def iter_header(i):
		s = "| Iteration "+str(i)+" |"
		d = "".join(["%" for c in s])
		return "\n" + d + "\n" + s + "\n" + d

out_files = {}
def write(fname, message, reset=False):
	global out_files
	if fname not in out_files: 
		out_files[fname] = open(fname,"w",buffering=1)
	f = out_files[fname]
	f.write(message+"\n")
	if reset:
		for fn in out_files:
			out_files[fn].close()
		out_files = {}

def find_saddle_point(
	cell,
	line_min=False,
	newton_raphson=False,
	max_step_size=0.05,
	force_tol=0.001,
	max_iter=100,
	minimizing=False):

	path = []                 # Will contain info about path
	success = False
	started_decent = False

	write(cell.seed+".out","%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	write(cell.seed+".out","| Begin saddle point search |")
	write(cell.seed+".out","|  (Activation-relaxation)  |")
	write(cell.seed+".out","%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
	if line_min: write(cell.seed+".out","Line minimization: on")
	else: write(cell.seed+".out","Line minimization: off")
	if newton_raphson: write(cell.seed+".out","Newton raphson: on")
	else: write(cell.seed+".out","Newton raphson: off")
	write(cell.seed+".out","Maximum normalized step size: "+str(max_step_size))
	write(cell.seed+".out","Force tolerance: "+str(force_tol)+" ev/ang")

	# Pick a random search direction
	rand = max_step_size*(np.random.rand(len(cell.config))*2-1)
	norm = rand/la.norm(rand)
	#if cell.test_potential: rand = MAX_STEP_SIZE * np.array([0,0.5])
	
	# Take initial random step
	cell.config += rand
	write(cell.seed+".out","\nInitial random step:")
	for i, par in enumerate(cell.variable_params()):
		write(cell.seed+".out","{0:20.20} += {1:10.5g}".format(par.name, rand[i]))

	for step_index in range(0, max_iter):

		write(cell.seed+".out", path_info.iter_header(step_index+1))
		p = path_info()
		p.path_name = "Saddle point search"

		# Initialize this step
		p.config = cell.config

		# Evaluate the force in the current 
		# configuration using finite differences
		p.pot, p.force  = cell.potential_and_force(fd_eps = max_step_size/10)
		write(cell.seed+".out", p.force_info(cell)+"\n")

		# Evaluate a normal and evaluate the 
		# parallel and perpendicular force
		# components to it
		p.norm  = p.config - cell.init_config
		p.norm /= la.norm(p.norm)
		p.fpara = np.dot(p.force, p.norm)*p.norm
		p.fperp = p.force - p.fpara

		if not started_decent and len(path) > 2:

			# Detect when we have left the basin
			if p.pot < path[-1].pot:
				write(cell.seed+".out","____________")		
				write(cell.seed+".out"," Left basin ")		
				write(cell.seed+".out","\_/^\_/^\_/^\n")		
				started_decent = True

		p.activation = np.zeros(len(p.config))
		p.relaxation = np.zeros(len(p.config))

		# Calculate activation step
		if started_decent and newton_raphson:

			p.path_name = "Saddle point search (newton raphson)"

			# Use newton-raphson to converge onto the saddle point
			new_config = np.zeros(len(p.config))
			for i in range(0, len(new_config)):
				denom = path[-1].force[i] - p.force[i]
				if denom == 0: ratio = 1
				else: ratio = p.force[i]/denom
				new_config[i] = p.config[i] + ratio * (p.config[i] - path[-1].config[i])

			# Clamp the move so we don't exceeed max_step_size
			disp = new_config - cell.config
			if la.norm(disp) > max_step_size:
				disp = max_step_size * disp / la.norm(disp)

			# Apply newton-raphson (as part of the relaxation step)
			p.relaxation += disp
		else:

			# Activate the configuration along the
			# normal by inverting the parallel force
			# component
			p.relaxation += max_step_size * p.fperp / la.norm(p.fperp) 
			sign = -1
			if minimizing: sign = 1
			p.activation += sign * max_step_size * p.fpara / la.norm(p.fpara)

		# Apply activation/relaxation step
		cell.config += p.activation
		cell.config += p.relaxation

		if line_min:

			# Perform a line minimization along the
			# perpendicular component
			cell.line_min_config(max_step_size, p.fperp, max_step_size*2)

		# Record this step
		write(cell.seed+".out", p.step_info(cell, la.norm(p.relaxation+p.activation)))
		write(cell.seed+".dat", p.verbose_info(cell))
		path.append(p)

		if la.norm(p.force) < force_tol:
			success = True
			write(cell.seed+".out",
			"\n|Force| = "+str(la.norm(p.force))+" < force_tol = "+str(force_tol))
			write(cell.seed+".out", "Saddle point reached")
			break

	return [success, path]

def find_minimum(cell, 
	init_direction, 
	max_step_size=0.05,
	force_tol=0.001):

	write(cell.seed+".out","\n%%%%%%%%%%%%%%%%%%%%%%")
	write(cell.seed+".out",  "| Begin minimization |")
	write(cell.seed+".out",  "%%%%%%%%%%%%%%%%%%%%%%")

	path = []
	init_config = cell.config
	cell.config += max_step_size * init_direction / la.norm(init_direction)
	close_to_minima = False

	for n in range(0,100):

		p = path_info()
		p.path_name = "Minimization"
		p.config = cell.config
		p.norm   = p.config - init_config 
		p.norm  /= la.norm(p.norm)

		write(cell.seed+".out", path_info.iter_header(n+1))

		p.pot, p.force = cell.potential_and_force()
		p.fpara  = np.dot(p.force, p.norm)*p.norm
		p.fperp  = p.force - p.fpara

		write(cell.seed+".out", p.force_info(cell)+"\n")
		
		if len(path) > 0:
			if p.pot > path[-1].pot:
				write(cell.seed+".out","_________________")		
				write(cell.seed+".out"," Close to minima ")		
				write(cell.seed+".out"," \_/ \_/ \_/ \_/ \n")		
				close_to_minima = True

		p.relaxation = np.zeros(len(p.config))
		p.activation = np.zeros(len(p.config))

		if close_to_minima and newton_raphson:
	
			p.path_name = "Minimization (newton raphson)"

			# Use newton-raphson to converge onto the minimum
			new_config = np.zeros(len(p.config))
			for i in range(0, len(new_config)):
				denom = path[-1].force[i] - p.force[i]
				if denom == 0: ratio = 1
				else: ratio = p.force[i]/denom
				new_config[i] = p.config[i] + ratio * (p.config[i] - path[-1].config[i])

			# Clamp the move so we don't exceeed max_step_size
			disp = new_config - cell.config
			if la.norm(disp) > max_step_size:
				disp = max_step_size * disp / la.norm(disp)

			# Apply newton-raphson (as part of the relaxation step)
			p.relaxation += disp

		else:
			# Steepest decent
			p.relaxation = p.force
			p.relaxation = max_step_size * p.relaxation / la.norm(p.relaxation)

		cell.config += p.relaxation

		write(cell.seed+".out", p.step_info(cell, la.norm(p.relaxation)))
		write(cell.seed+".dat", p.verbose_info(cell))
		path.append(p)

		if la.norm(p.force) < force_tol:
			write(cell.seed+".out",
			"\n|Force| = "+str(la.norm(p.force))+" < force_tol = "+str(force_tol))
			write(cell.seed+".out", "Minimum reached")
			break

	return path

def plot_path_info(path, cell, plot_pot=True):

	# Plot information about a path
	# through the configuration space 
	# of the given cell

	scales = [par.scale for par in cell.variable_params()]

	plt.subplot(1,2,1)
	if cell.test_potential and plot_pot: cell.plot_potential()
	plt.plot([p.config[0]*scales[0] for p in path],
	         [p.config[1]*scales[1] for p in path], marker="+")

	if False:
		for p in path: 
			cf = p.config * scales
			ac = p.activation   * scales
			lm = p.relaxation * scales
			plt.plot(*zip(*[cf, cf + ac]), color="red")
			plt.plot(*zip(*[cf + ac, cf + ac + lm]), color="blue")
	
	plt.subplot(4,4,3)
	plt.plot([p.pot for p in path])
	plt.xlabel("Iteration")
	plt.ylabel("Potential")

	plt.subplot(4,4,4)
	plt.plot([la.norm(p.force) for p in path])
	plt.xlabel("Iteration")
	plt.ylabel("|Force|")

	plt.subplot(4,4,7)
	plt.plot([la.norm(path[i].config-path[i-1].config) for i in range(1,len(path))])
	plt.xlabel("Iteration")
	plt.ylabel("Step size\n(Normalized)")

	plt.subplot(4,4,8)
	plt.plot([la.norm(p.config-cell.init_config) for p in path])
	plt.xlabel("Iteration")
	plt.ylabel("Normalized displacement")

#########################
#      Run program      #
#########################

# Clean up directory structure
os.system("rm -r singlepoints")
os.system("mkdir singlepoints")

# Set global variables
castep_cmd = "nice -15 castep.threaded"
cell = cell(sys.argv[1]+".cell")
line_min = False
newton_raphson = False
max_step_size = 0.05
force_tol = 0.001
max_iter = 100

# Parse the .saddle file for input specification
lines = [l.strip().lower() for l in open(sys.argv[1]+".saddle").read().split("\n")]
for l in lines:

	# Ignore empty lines
	if len(l) == 0: continue

	# Ignore comment lines
	if l.startswith("#"): continue
	if l.startswith("!"): continue
	if l.startswith("//"): continue

	# Parse line into form "tag vals[0] vals[1] vals[2]..."
	tag  = l.split()[0]
	vals = l.split()[1:]

	if tag == "fix_atoms":
		# Fix the atoms specified (or all if none specified explicitly)
		if len(vals) > 0: cell.fix_atoms([int(w) for w in vals])
		else: cell.fix_atoms()

	elif tag == "fix_lattice_params":
		# Fix the lattice params specified (or all if none specified explicitly)
		if len(vals) > 0: cell.fix_lattice_params("".join(vals))
		else: cell.fix_lattice_params("ABC")

	elif tag == "fix_lattice_angles":
		# Fix the lattice angles specified (or all if none specified explicitly)
		if len(vals) > 0: cell.fix_lattice_angles("".join(vals))
		else: cell.fix_lattice_angles("ABC")

	elif tag == "test":
		# Turn on the test potential (rather than using castep)
		cell.test_potential = True

	elif tag == "line_minimize":
		# Turn on line minimization"
		line_min = True

	elif tag == "newton_raphson":
		# Turn on newton raphson convergence
		newton_raphson = True

	elif tag == "max_step_size":
		# Read in maximum step size
		max_step_size = float(vals[0])

	elif tag == "force_tol":
		# Read in force tolerance
		force_tol = float(vals[0])

	elif tag == "max_iter":
		# Read in maximum iterations
		max_iter = int(vals[0])

# Set up the cell correctly if we're using the test potential
if cell.test_potential: 
	cell.unfix_all()
	cell.fix_atoms()
	cell.fix_lattice_params("C")
	cell.fix_lattice_angles("ABC")

# Output information on parameters in the cell
write(cell.seed+".out", "Cell parameters")
if cell.test_potential:
	write(cell.seed+".out", "(cell is running in test mode, with fake potentials/forces)")
titles = "{0:20.20}  {1:10.10}  {2:10.10}  {3:5.5}".format("Parameter","Value","Scale","Fixed")
div    = "".join(["-" for c in titles])
write(cell.seed+".out", div+"\n"+titles+"\n"+div)
for p in cell.params:
	fs = "{0:20.20}  {1:10.5g}  {2:10.5g}  {3:5}"
	write(cell.seed+".out", fs.format(p.name, p.value, p.scale, p.fixed))
write(cell.seed+".out", "")

if not cell.test_potential:
	# Run the saddle point algorithm
	suc, path = find_saddle_point(cell, line_min=line_min, max_step_size=max_step_size,
	 			      newton_raphson=newton_raphson, force_tol=force_tol,
				      max_iter=max_iter)

	if suc: 
		write(cell.seed+".out", "\n=================================")
		write(cell.seed+".out",   "| Success: Saddle point reached |")
		write(cell.seed+".out",   "=================================")

		find_minimum(cell, cell.config - cell.init_config)
	else:
		write(cell.seed+".out", "\n=======================================")
		write(cell.seed+".out",   "| Failed: maximum iterations reached! |")
		write(cell.seed+".out",   "=======================================")
		
else:
	# Run test several times
	repeats = 10
	successes = 0
	av_singlepoints = 0
	for n in range(0,repeats):

		cell.reset()

		suc, path = find_saddle_point(cell, line_min=line_min, max_step_size=max_step_size,
					      newton_raphson=newton_raphson, force_tol=force_tol,
					      max_iter=max_iter)

		av_singlepoints += cell.singlepoint_count

		if suc: 
			successes += 1
			path.extend(find_minimum(cell, cell.config - cell.init_config,
				    max_step_size=max_step_size, force_tol=force_tol))

		plot_path_info(path, cell, plot_pot=n==0)
		write(cell.seed+".out","",reset=True)

	print "Average singlepoint evaluations: ", float(av_singlepoints)/repeats
	print "Success rate: ", 100*(float(successes)/repeats), "%"
	plt.show()
