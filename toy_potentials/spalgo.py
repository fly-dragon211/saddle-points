import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import potentials as pots

def act_relax(grad, max_range=np.inf, max_step=0.1, max_iter=100):
	
	# Initial random move
	norm  = 2*(np.random.rand(2)-0.5)
	norm /= la.norm(norm)
	path  = [np.array(norm/100)]

	for n in range(0, max_iter):
		
		# Evaluate force and normal at path[-1]
		f     = -grad(path[-1])
		norm  = path[-1]/la.norm(path[-1])

		# Get parallel and perp compoenents of force
		fpara = np.dot(norm,f)*norm
		fperp = f - fpara

		# Move along perpendicular and against parallel
		delta = fperp - fpara

		# Enforce max step size
		if la.norm(delta) > max_step:
			delta = max_step * delta/la.norm(delta)

		path.append(path[-1] + delta) 

		if la.norm(f) < 0.001: break
		if la.norm(path[-1]) > max_range: break

	return path

def act_relax_fd(f, max_range=np.inf, max_step=0.1, max_iter=100):
	g = lambda x : pots.grad_fd(x, f)
	return act_relax(g, max_range=max_range, max_step=max_step, max_iter=max_iter)

def min_mode(grad,hess):
	
	# Initial random move
	norm  = 2*(np.random.rand(2)-0.5)
	norm /= la.norm(norm)
	path  = [np.array(norm)/10]

	for n in range(0,100):
		
		h = hess(path[-1])
		g = grad(path[-1])

		evals, evecs = la.eig(h)

		both = zip(evals, evecs)
		both.sort()
		evals, evecs = zip(*both)

		dots = [np.abs(np.dot(e, path[-1])) for e in evecs]
		imax = dots.index(max(dots))

		delta = np.zeros(len(path[-1]))
		for i in range(0,len(evecs)):

			# Change the sign of the step
			# along the minimum mode
			sign = 1
			if i == 0 or i == imax: sign = -1

			# Move according to newton raphson
			if abs(evals[i]) > 10e-6:
				delta -= evecs[i] * np.dot(g, evecs[i]) * sign / evals[i]

		# Impose max step size
		if la.norm(delta) > 0.1:
			delta = 0.1 * delta / la.norm(delta)

		path.append(path[-1] + delta)
		if la.norm(g) < 0.01:
			break

	return path

def min_mode_fd(f):
	
	g = lambda x : pots.grad_fd(x, f)
	h = lambda x : pots.hess_fd(x, f)
	return min_mode(g,h)
