from scipy.optimize import minimize as scipy_min
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import potentials as pots

def act_relax(pot, grad, hess,
	start=[0,0], max_range=np.inf, max_step=0.1, max_iter=100, force_tol=0.001):
	
	# Initial random move
	start = np.array(start)
	norm  = 2*(np.random.rand(len(start))-0.5)
	norm /= la.norm(norm)
	path  = [start + np.array(norm/100)]
	
	forces = []

	for n in range(0, max_iter):
		
		# Evaluate force
		f = -grad(path[-1])
		forces.append(f)

		# Get parallel and perp compoenents of force
		norm  = path[-1]-start
		norm /= la.norm(norm)
		fpara = np.dot(norm,f)*norm
		fperp = f - fpara

		# Move along perpendicular and against parallel
		delta = max_step*(fperp - fpara)/np.mean([la.norm(f) for f in forces])

		# Enforce max step size
		if la.norm(delta) > max_step:
			delta = max_step * delta/la.norm(delta)

		path.append(path[-1]+delta)

		if la.norm(f) < force_tol: break
		if la.norm(path[-1]) > max_range: break

	return path

def taylor(r, r0, v0, force, hess):
	dr = r - r0
	return v0 - np.dot(dr, force) + 0.5 * np.dot(dr.T, np.dot(hess, dr))

def hess_obj(x, potential, force, positions, potentials, hess):
	
	cost = 0
	for i, p in enumerate(positions):
		dp  = np.array(p)-np.array(x)
		pot = taylor(p, x, potential, force, hess)
		cost += ((pot - potentials[i])**2)#*np.exp(-la.norm(dp))
	return cost

def unflatten_hessian(h, n):
	ret = np.zeros((n,n))
	for x in range(0, n):
		for y in range(0, n):
			ret[x][y] = h[x+y*n]
	return ret

def fit_hess(x, potential, force, positions, potentials):
	to_min = lambda h : hess_obj(x, potential, force, positions, potentials, unflatten_hessian(h,len(x)))
	hess = unflatten_hessian(scipy_min(to_min, np.identity(len(x))).x, len(x))
	return hess

def test_hess_obj():
	
	for m in range(0,10):
		
		r = lambda : np.random.rand()*2-1
		hess = [[r(),r()],[r(),r()]]

		rs = []
		fs = []
		force = [r(),r()]
		pot = r()
		origin = [r(),r()]

		for n in range(0,4):

			r = np.array(np.random.rand(2))
			rs.append(r)
			fs.append(taylor(r, origin, pot, force, hess))

		hess_fit = fit_hess(origin, pot, force, rs, fs)
		print hess_fit - hess
		print hess_fit
		print ""

#test_hess_obj()
#quit()

def newton_step(force, hessian):
	
	evals, evecs = la.eig(hessian)
	imin = list(evals).index(min(evals))
	step = np.zeros(len(force))
	for i in range(0, len(evecs)):
		sign = 1 if i == imin else -1
		step += sign * evecs[i] * np.dot(evecs[i], -force) / evals[i]
	return step


def delta_sq(lamb, evals, evecs, grad):

	tot = 0
	for i in range(0, len(evals)):
		di = np.dot(grad, evecs[i])
		tot += di**2/(lamb - evals[i])**2
	return tot

def best_lamb(hess, grad):

	evals, evecs = la.eig(hess)
	both = zip(evals, evecs)
	both.sort(lambda l1, l2: 1 if l1[0] > l2[0] else -1)
	evals, evecs = zip(*both)
	if evals[0] < 0: return 0

	to_min = lambda l : delta_sq(l, evals, evecs, grad)
	x0 = [0.5*evals[0]+0.5*evals[1]]
	bs = [(evals[0], evals[1])]
	min_res = scipy_min(to_min, x0, bounds=bs)

	return min_res.x

def min_mode(pot, grad, hess,
	start=[0,0], max_range=np.inf, max_step=0.1, max_iter=100, force_tol=0.001):

	start = np.array(start)
	norm  = 2*(np.random.rand(len(start))-0.5)
	norm /= la.norm(norm)
	path  = [start+norm*max_step]

	for n in range(0, max_iter):
		
		h = hess(path[-1])
		g = grad(path[-1])

		lamb = best_lamb(h,g)
		dx = np.dot(la.inv(np.identity(len(g))*lamb - h), g)

		if la.norm(dx) > max_step:
			dx = max_step * dx / la.norm(dx)

		path.append(path[-1] + dx)

		if la.norm(g) < force_tol: break
		if la.norm(path[-1]) > max_range: break

	return path

def act_relax_ah(pot, grad, hess,
	start=[0,0], max_range=np.inf, max_step=0.1, max_iter=100, force_tol=0.001):
	
	# Initial random move
	start = np.array(start)
	norm  = 2*(np.random.rand(len(start))-0.5)
	norm /= la.norm(norm)
	path  = [start + np.array(norm/100)]

	forces     = []
	positions  = []
	potentials = []

	for n in range(0, max_iter):
		
		# Evaluate force
		f     = -grad(path[-1])
		forces.append(f)
		positions.append(path[-1])
		potentials.append(pot(path[-1]))

		# Get parallel and perp compoenents of force
		norm  = path[-1]-start
		norm /= la.norm(norm)
		fpara = np.dot(norm,f)*norm
		fperp = f - fpara

		# Move along perpendicular and against parallel
		delta = max_step*(fperp - fpara)/np.mean([la.norm(fi) for fi in forces])
		if la.norm(delta) > max_step:
			delta = max_step * delta / la.norm(delta)

		# Fit an approximate hessian to nearby points
		pos_to_fit = []
		pot_to_fit = []
		for i, p in enumerate(positions):
			if la.norm(p - path[-1]) >= max_step*2: continue
			pos_to_fit.append(p)
			pot_to_fit.append(potentials[i])

		# If we have enough points to sensibly construct a hessian
		if len(pos_to_fit) >= 2:

			hess_here = fit_hess(path[-1], pot(path[-1]), f, pos_to_fit, pot_to_fit)
			pots.pot_evals -= 1 # In a real calc we would get potential for free with the force

			# Calculate a minimum-mode step using the approximate hessian
			lamb = best_lamb(hess_here,-f)
			rescaled = np.identity(len(f))*lamb - hess_here
			try: 
				nr_step = np.dot(la.inv(rescaled), -f) - delta
				if la.norm(nr_step) > max_step:
					nr_step *= max_step/la.norm(nr_step)
			except: 
				nr_step = np.zeros(len(f))
		else:
			nr_step = np.zeros(len(f))

		# Take the step
		path.append(path[-1] + delta)
		path.append(path[-1] + nr_step*la.norm(delta)/2)

		# Check convergence
		if la.norm(f) < force_tol: break
		if la.norm(path[-1]) > max_range: break

	print len(path)
	return path

def steepest_acent(pot, grad, hess,
	start=[0,0], max_range=np.inf, max_step=0.1, max_iter=100, force_tol=0.001):
	
	start = np.array(start)
	norm  = 2*(np.random.rand(len(start))-0.5)
	norm /= la.norm(norm)
	path  = [np.array(norm)/10]

	for n in range(0, max_iter):
		
		g = grad(path[-1])
		
		delta = g
		if la.norm(delta) > max_step:
			delta = max_step * delta / la.norm(delta)

		path.append(path[-1] + delta)

		if la.norm(g) < force_tol: break

	return path

def min_det(pot, grad, hess,
	start=[0,0], max_range=np.inf, max_step=0.1, max_iter=100, force_tol=0.001):

	path = [start]
	f = lambda x : la.det(hess(x))
	c = lambda x : path.append(x)
	scipy_min(f, path[-1], callback=c, method="bfgs",
		options={"eps":0.01, "gtol":1.0})
	return path

methods = [act_relax, min_mode, act_relax_ah]
