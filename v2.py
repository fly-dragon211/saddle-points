import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

DX_MAX     = 0.1
EPS        = DX_MAX / 10
GRAD_THR   = 0.01
SCALE_FACT = 2

sort_point = None
def sort_closest(a,b):
	if la.norm(a[0] - sort_point) < la.norm(b[0] - sort_point):
		return -1
	else:
		return 1

data_real_grid = None
real_grid_centre = None
def pot_real_grid(c, direc):
	global data_real_grid, real_grid_max, real_grid_centre

	if data_real_grid == None:
		data_real_grid = []
		e_min = np.inf
		for f in os.listdir("./real_potentials/"+direc):
			if not f.endswith(".castep"): continue
			seed = f[0:-len(".castep")]
			splt = seed.split("_")
			if len(splt) != 2: continue
			fx = float(splt[0])
			fy = float(splt[1])
			lines = open("./real_potentials/"+direc+"/"+f).read().split("\n")
			e = None
			for l in lines:
				if "NB" in l:
					e = float(l.split("=")[1].split("e")[0])
					break
			if e == None: continue
			if e < e_min:
				e_min = e
				real_grid_centre = np.array([fx,fy])
			data_real_grid.append([np.array([fx,fy]),e])
		print "Read "+str(len(data_real_grid))+" points on the potential at "+direc

	c = np.array(c) + real_grid_centre

	p = 4
	ret = 0
	renorm = 0
	for pt in data_real_grid:
		dis = la.norm(c-pt[0])**p
		if dis == 0: return pt[1]
		ret += pt[1] / dis
		renorm += 1 / dis
	return ret/renorm

def pot_li_bcc_fcc(c):
	return pot_real_grid(c,"li_bcc_fcc")

def pot_fe_bcc_fcc(c):
	return pot_real_grid(c, "fe_bcc_fcc")

def pot_fe_bcc_fcc_dense(c):
	return pot_real_grid(c, "fe_bcc_fcc_dense")

def pot_egg(c):
	x = c[0]
	y = c[1]
	return np.sin((x-0.5)*np.pi)*np.cos(y*np.pi)

def grad_egg(c):
	x = c[0]
	y = c[1]
	gx =  np.pi*np.cos((x-0.5)*np.pi)*np.cos(np.pi*y)
	gy = -np.pi*np.sin((x-0.5)*np.pi)*np.sin(np.pi*y)
	return np.array([gx,gy])

def hess_egg(c):
	x = c[0]
	y = c[1]
	xx = -np.pi*np.pi*np.sin((x-0.5)*np.pi)*np.cos(y*np.pi)
	xy = -np.pi*np.pi*np.cos((x-0.5)*np.pi)*np.sin(y*np.pi)
	return np.array([[xx,xy],[xy,xx]])

def pot_coulomb(c):
	ret = 0
	for x in np.arange(-2.5,2.6,1):
		for y in np.arange(-2.5,2.6,1):
			ret += 1/la.norm(c-np.array([x,y]))
	return ret

def pot_coulomb_twist(c):
	r = la.norm(c)
	sin = np.sin(r)
	cos = np.cos(r)
	rot = np.array([[sin,cos],[cos,-sin]])
	return pot_coulomb(np.matmul(rot,c))

selected_potential = pot_egg
pot_evals = 0
def pot(c):
	global pot_evals
	pot_evals += 1
	return selected_potential(c)

def grad(c):
	global EPS
	ret = np.zeros(len(c))
	p0 = pot(c)
	for i in range(0,len(c)):
		di = np.zeros(len(c))
		di[i] = EPS
		ret[i] = pot(c+di)-p0
		ret[i] /= EPS
	return ret

def hess(c):
	global EPS
	ret = np.zeros([len(c),len(c)])
	p0 = pot(c)
	for i in range(0,len(c)):
		for j in range(0,len(c)):
			di = np.zeros(len(c))
			dj = np.zeros(len(c))
			di[i] = EPS
			dj[j] = EPS
			hij = pot(c+di+dj)-pot(c+di-dj)-pot(c-di+dj)+pot(c-di-dj)
			hij /= 4*EPS**2
			ret[i][j] = hij
	return ret

def plot_potential():
        RANGE = 2
        RESOL = 100
        x   = np.linspace(-RANGE,RANGE,RESOL)
        y   = np.linspace(-RANGE,RANGE,RESOL)
        z   = []
        all_vals = []
	max_val = -np.inf
	min_val = np.inf
        for yi in y:
                row = []
                for xi in x:
                        p = np.array([xi,yi])
                        val = pot(p)
                        row.append(val)
                        all_vals.append(val)
			if val > max_val : max_val = val
			if val < min_val : min_val = val
                z.append(row)
	z = ((z-min_val)/(max_val-min_val))**(0.25)
        x,y = np.meshgrid(x,y)
	levels = np.linspace(0,1,30)
        plt.contour(x,y,z,cmap="tab20c",levels=levels)
        plt.imshow(z,cmap="tab20c",extent=(-RANGE,RANGE,-RANGE,RANGE),
		   origin="lower",interpolation="bilinear",alpha=0.2)
	plt.xlabel("x")
	plt.ylabel("y")

def simple_climb():
	path = []
	x = np.zeros(2) + 0.01*(np.random.rand(2)*2-1)
	dx_scale = 1.0
	for n in range(0,100):
		xh = x/la.norm(x)
		f  = -grad(x)
		if la.norm(f) < GRAD_THR:
			break
		fpara = np.dot(f,xh)*xh
		fperp = f - fpara

		dx = fperp - fpara
		x += dx_scale*DX_MAX*dx/la.norm(dx)

		path.append(np.array(x))

		if len(path) > 3:
			if np.dot(path[-1]-path[-2],path[-2]-path[-3]) < 0:
				dx_scale /= SCALE_FACT

	return path

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

def interp(xs, x1, x2, f1, f2, g1, g2):
        m = np.array([
        [1,x1,x1**2,x1**3],
        [1,x2,x2**2,x2**3],
        [0,1,2*x1,3*x1**2],
        [0,1,2*x2,3*x2**2]
        ])
        try:
                minv = la.inv(m)
        except:
                print "Singular matrix!"
                print m
                quit()
        b = [f1,f2,g1,g2]
        a = np.matmul(minv,b)
        return [a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3 for x in xs]

pot_evals_line_min = 0
def line_minimize(x_start, direction):
	global pot_evals_line_min

        step = 0.01
        d = direction/la.norm(direction)

        mapped_pos = [x_start]
	pot_evals_line_min += 1
        mapped_pot = [pot(x_start)]
        mapped_del = [0]
        first = mapped_pot[0]

        while True:
                min_index = mapped_pot.index(min(mapped_pot))
                if min_index == 0:
                        xtest = mapped_pos[0]-step*d
                        mapped_pos.insert(0, xtest)
			pot_evals_line_min += 1
                        mapped_pot.insert(0, pot(xtest))
                        mapped_del.insert(0, mapped_del[0]-step)
                elif min_index == len(mapped_pot)-1:
                        xtest = mapped_pos[-1]+step*d
                        mapped_pos.append(xtest)
			pot_evals_line_min += 1
                        mapped_pot.append(pot(xtest))
                        mapped_del.append(mapped_del[-1]+step)
                else:
                        break

	xs = np.linspace(min(mapped_del)+step/100, max(mapped_del)-step/100, 100)
	ys = interp_new(xs, mapped_del, mapped_pot)
	return x_start + d*xs[list(ys).index(min(ys))]


def simple_climb_line_min():
	path = [np.zeros(2)]
	x = np.zeros(2)
	dx_init = np.random.rand(2)*2-1
	dx_init = DX_MAX * dx_init / la.norm(dx_init)
	x = x + dx_init
	path.append(x.copy())
	dx_scale = 1.0
	for step_index in range(0,200):

		f  = -grad(x)

		if la.norm(f) < GRAD_THR:
			break

		n = x/la.norm(x)
		fpara = np.dot(f,n)*n
		fperp = f - fpara

		dx = fperp - fpara
		dx /= la.norm(dx)

		if len(path) > 2:
			if np.dot(path[-1]-path[-2], dx) < 0:
				dx_scale /= SCALE_FACT

		dx /= la.norm(dx)
		dx *= dx_scale * DX_MAX

		x += dx
		x = line_minimize(x, fperp)

		path.append(np.array(x))
	return path

def support_points():
	path = [np.zeros(2)]
	rand = np.random.rand(2)*2-1
	rand = DX_MAX * rand / la.norm(rand)
	path.append(path[-1].copy() + rand)
	d = len(path[-1])
	
	for step_index in range(0,100):
		
		pot_here = pot(path[-1])
		delta_interp = DX_MAX
		grad = np.zeros(d)

		for i in range(0,d):
			di = np.zeros(d)
			di[i] = delta_interp
			pot_plus  = pot(path[-1]+di)
			pot_minus = pot(path[-1]-di)
			grad[i] = (pot_plus - pot_minus)/(2*delta_interp)

		f = - grad
		n = path[-1].copy()/la.norm(path[-1])			
		fpara = np.dot(f,n)*n
		fperp = f - fpara
		dx = fperp - fpara
		dx = DX_MAX * dx / la.norm(dx)
		path.append(path[-1].copy() + dx)

	return path

def art_n():
	path = [np.zeros(2)]
	rand = np.random.rand(2)*2-1
	rand = DX_MAX * rand / la.norm(rand)
	path.append(path[-1].copy() + rand)

	neg_eval_dir = None

	for step_index in range(0, 100):

		f = - grad(path[-1].copy())
		if la.norm(f) < GRAD_THR: break

		n = path[-1].copy()
		n /= la.norm(n)

		if neg_eval_dir is None:
			h = hess(path[-1])
			evals, evecs = la.eig(h)
			i_min = list(evals).index(min(evals))
			if evals[i_min] < 0:
				neg_eval_dir = evecs[i_min]	
		else:
			n = neg_eval_dir

		fpara = np.dot(f,n)*n
		fperp = f - fpara
		dx = fperp - fpara
		dx = DX_MAX * dx / la.norm(dx)
		if la.norm(dx) > DX_MAX + EPS: print "Error |dx| > DX_MAX: ", la.norm(dx), " > ", DX_MAX
		path.append(path[-1].copy() + dx)
		path.append(line_minimize(path[-1].copy(), fperp))

	return path

def rfo():
	path = []
	x = np.zeros(2) + DX_MAX*(np.random.rand(2)*2-1)
	for n in range(0,1000):
		g = grad(x)
		h = hess(x)
		l, v = la.eig(h)
		min_i = list(l).index(min(l))

		if l[min_i] > 0:
			dx = DX_MAX*x/la.norm(x)
		else:
			dx = np.zeros(2)
			for i in range(0,len(v)):
				gi = np.dot(g,v[i])
				d = 0.5 * (abs(l[i]) - np.sqrt(l[i]**2+4*gi**2))
				if i == min_i: d = -d
				if d == 0: d = 0.0001
				dx += -gi*v[i]/d

			if la.norm(dx) > DX_MAX:
				dx = DX_MAX*dx/la.norm(dx)

			if la.norm(g) < 1:
				dx *= la.norm(g)
		x += dx
		path.append(np.array(x))

		if la.norm(g) < GRAD_THR:
			break

	return path

def min_mode():
	path = []
	x = np.zeros(2) + DX_MAX*(np.random.rand(2)*2-1)
	dx_scale = 1.0
	for n in range(0,1000):
		g = grad(x)
		if la.norm(g) < GRAD_THR:
			break

		h = hess(x)
		l, v = la.eig(h)
		min_i = list(l).index(min(l))

		if l[min_i] >= 0:
			dx = DX_MAX*x/la.norm(x)
		else:
			f = -g
			fpara = np.dot(f, v[min_i]) * v[min_i]
			fperp = f-fpara
			dx = fperp - fpara
			if la.norm(dx) > DX_MAX:
				dx = DX_MAX*dx/la.norm(dx)
		x += dx_scale*dx
		path.append(np.array(x))

		if len(path) > 3:
			if np.dot(path[-1]-path[-2],path[-2]-path[-3]) < 0:
				dx_scale /= SCALE_FACT

		if n == 999:
			path = []
			break

	return path

selected_potential = pot_li_bcc_fcc
selected_potential = pot_egg
selected_potential = pot_fe_bcc_fcc
selected_potential = pot_fe_bcc_fcc_dense
selected_potential = pot_coulomb
selected_potential = pot_coulomb_twist

saddle_method = rfo
saddle_method = min_mode
saddle_method = simple_climb
saddle_method = support_points
saddle_method = art_n
saddle_method = simple_climb_line_min

def opt_param():
	global pot_evals, SCALE_FACT, DX_MAX

	dx_max_before = DX_MAX
	xs  = np.linspace(0.01, 0.31, 20)
	ys  = []
	dys = []
	for DX_MAX in xs:
		pevs = []
		for n in range(0,100):
			pot_evals = 0
			path = saddle_method()
			pevs.append(pot_evals)
		ys.append(np.mean(pevs))
		dys.append(np.sqrt(np.var(pevs)))
	
	ys = np.array(ys)
	dys = np.array(dys)
	plt.plot(xs,ys)
	plt.fill_between(xs, ys+dys, ys-dys, alpha = 0.2)
	plt.xlabel("max(|dx|)")
	plt.ylabel("Average potential evaluations")
	plt.show()
	DX_MAX = dx_max_before

	scale_fact_before = SCALE_FACT
	xs  = np.linspace(1,10,100)
	ys  = []
	dys = []
	for SCALE_FACT in xs:
		pevs = []
		plen = []
		for n in range(0,10):
			pot_evals = 0
			path = saddle_method()
			pevs.append(pot_evals)
		
		ys.append(np.mean(pevs))
		dys.append(np.sqrt(np.var(pevs)))

	ys = np.array(ys)
	dys = np.array(dys)
	plt.plot(xs,ys)
	plt.fill_between(xs, ys-dys, ys+dys, alpha=0.2)
	plt.xlabel("Scale reduction factor")
	plt.ylabel("Average potential evaluations")
	plt.show()
	SCALE_FACT = scale_fact_before

def plot_method(repeats=100):

	spx = 4
	spy = 4

	global pot_evals, pot_evals_line_min
	pot_axis = plt.subplot(1, 2, 1)
	plot_potential()

	for n in range(0,repeats):

		print "Method          : ", saddle_method
		print "Potential       : ", selected_potential

		pot_evals = 0
		pot_evals_line_min = 0
		path = saddle_method()

		print "Steps           : ", len(path)
		print "Potential evals : ", pot_evals
		print "    Line min pe : ", pot_evals_line_min
		print "Final coord     : ", path[-1]
		print "Hessian eigvals : ", la.eig(hess(path[-1]))[0]
		sfstring = "Fail"
		if la.norm(grad(path[-1])) < GRAD_THR: sfstring = "Success"
		print "Gradient        : ", grad(path[-1]), sfstring

		pot_axis.plot(*zip(*path), marker="+")

		eigs = [min(la.eig(hess(p))[0]) for p in path]
		plt.subplot(spy, spx, 3)
		plt.plot(eigs)
		plt.axhline(0, color="red")
		plt.xlabel("Iteration")
		plt.ylabel("Min eigenvalue")

		ss = [la.norm(path[i]-path[i-1]) for i in range(1,len(path))]
		plt.subplot(spy, spx, 4)
		plt.plot(ss)
		plt.xlabel("Iteration")
		plt.ylabel("Step size")
		plt.axhline(DX_MAX, color="red")

		grads = [la.norm(grad(p)) for p in path]
		plt.subplot(spy, spx, 7)
		plt.plot(grads)
		plt.xlabel("Iteration")
		plt.ylabel("|grad|")

		plt.subplot(spy, spx, 8)
		pots = [pot(p) for p in path]
		plt.xlabel("Iteration")
		plt.ylabel("Potential")
		plt.plot(pots)

		origin_dist = [la.norm(p) for p in path]
		plt.subplot(spy, spx, 11)
		plt.xlabel("Iteration")
		plt.ylabel("Distance from origin")
		plt.plot(origin_dist)

		plt.subplot(spy, spx, 12)
		plt.plot(origin_dist, pots, linestyle="none", marker="+")
		plt.xlabel("Distance from origin")
		plt.ylabel("Potential")
		#xs = np.linspace(min(origin_dist)+10e-6, max(origin_dist)-10e-6, 100)
		#ys = interp_new(xs, origin_dist, pots)
		#plt.plot(xs, ys)

	plt.show()

plot_method(repeats=1)
