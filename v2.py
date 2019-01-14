import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

EPS        = 0.0001
GRAD_THR   = 0.1
DX_MAX     = 0.1
SCALE_FACT = 2

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
	xs = [0.5,0.5,-0.5,-0.5]
	ys = [0.5,-0.5,0.5,-0.5]
	ret = 0
	for x in xs:
		for y in ys:
			ret += 1/la.norm(c-np.array([x,y]))
	return ret

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
        RESOL = 50
        x   = np.linspace(-RANGE,RANGE,RESOL)
        y   = np.linspace(-RANGE,RANGE,RESOL)
        z   = []
        all_vals = []
        for yi in y:
                row = []
                for xi in x:
                        p = np.array([xi,yi])
                        val = pot(p)
                        row.append(val)
                        all_vals.append(val)
                z.append(row)
        x,y = np.meshgrid(x,y)
        plt.contour(x,y,z,cmap="seismic")
        plt.imshow(z,cmap="seismic",extent=(-RANGE,RANGE,-RANGE,RANGE),
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
        coeff = np.matmul(minv, b)

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

        success = True
        try:
		xs = np.linspace(min(mapped_del)+step/100, max(mapped_del)-step/100, 100)
		ys = interp_new(xs, mapped_del, mapped_pot)
		return x_start + d*xs[list(ys).index(min(ys))]

		plt.plot(mapped_del, mapped_pot, marker="+", linestyle="none")
		plt.plot(xs,ys)
		plt.show()
                par, covar = curve_fit(line_min_fit, mapped_del, mapped_pot, p0=[min(mapped_pot), 0, 1])
		plt.plot(mapped_del, line_min_fit(mapped_del, *par))
        except:
                success = False
		print "Line min failed"

        if success:
                return par[1]*d + x_start
        else:
                return mapped_pos[mapped_pot.index(min(mapped_pot))]


def simple_climb_line_min():
	path = [np.zeros(2)]
	x = np.zeros(2) + 0.01*(np.random.rand(2)*2-1)
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

		x += dx_scale*DX_MAX*dx/la.norm(dx)
		x = line_minimize(x, fperp)

		path.append(np.array(x))

		if False:
			if len(path) > 3:
				if np.dot(path[-1]-path[-2],path[-2]-path[-3]) < 0:
					dx_scale /= SCALE_FACT
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

selected_potential = pot_egg
selected_potential = pot_coulomb

saddle_method = rfo
saddle_method = min_mode
saddle_method = simple_climb
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

	spx = 3
	spy = 3

	global pot_evals, pot_evals_line_min
	plt.subplot(spy, spx, 1)
	plot_potential()

	for n in range(0,repeats):

		pot_evals = 0
		pot_evals_line_min = 0
		path = saddle_method()

		print "Method          : ", saddle_method
		print "Potential       : ", selected_potential
		print "Steps           : ", len(path)
		print "Potential evals : ", pot_evals
		print "    Line min pe : ", pot_evals_line_min
		print "Final coord     : ", path[-1]
		print "Hessian eigvals : ", la.eig(hess(path[-1]))[0]
		sfstring = "Fail"
		if la.norm(grad(path[-1])) < GRAD_THR: sfstring = "Success"
		print "Gradient        : ", grad(path[-1]), sfstring

		plt.subplot(spy, spx, 1)
		plt.plot(*zip(*path), marker="+")

		eigs = [min(la.eig(hess(p))[0]) for p in path]
		plt.subplot(spy, spx, 2)
		plt.plot(eigs)
		plt.axhline(0, color="red")
		plt.xlabel("Iteration")
		plt.ylabel("Min eigenvalue")

		ss = [la.norm(path[i]-path[i-1]) for i in range(1,len(path))]
		plt.subplot(spy, spx, 3)
		plt.plot(ss)
		plt.xlabel("Iteration")
		plt.ylabel("Step size")
		plt.axhline(DX_MAX, color="red")

		grads = [la.norm(grad(p)) for p in path]
		plt.subplot(spy, spx, 4)
		plt.plot(grads)
		plt.xlabel("Iteration")
		plt.ylabel("|grad|")

		plt.subplot(spy, spx, 5)
		pots = [pot(p) for p in path]
		plt.xlabel("Iteration")
		plt.ylabel("Potential")
		plt.plot(pots)

		origin_dist = [la.norm(p) for p in path]
		plt.subplot(spy, spx, 6)
		plt.xlabel("Iteration")
		plt.ylabel("Distance from origin")
		plt.plot(origin_dist)

		plt.subplot(spy, spx, 7)
		plt.plot(origin_dist, pots, linestyle="none", marker="+")
		plt.xlabel("Distance from origin")
		plt.ylabel("Potential")
		#xs = np.linspace(min(origin_dist)+10e-6, max(origin_dist)-10e-6, 100)
		#ys = interp_new(xs, origin_dist, pots)
		#plt.plot(xs, ys)

	plt.show()

plot_method(repeats=100)
