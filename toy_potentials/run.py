import potentials as pots
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

def act_relax():
	
	# Initial random move
	norm  = 2*(np.random.rand(2)-0.5)
	norm /= la.norm(norm)
	path  = [np.array(norm/100)]

	for n in range(0,100):
		
		# Evaluate force and normal at path[-1]
		f     = -pots.grad(path[-1])
		norm  = path[-1]/la.norm(path[-1])

		# Get parallel and perp compoenents of force
		fpara = np.dot(norm,f)*norm
		fperp = f - fpara

		# Move along perpendicular and against parallel
		delta = fperp - fpara

		# Enforce max step size
		if la.norm(delta) > 0.1:
			delta = 0.1 * delta/la.norm(delta)

		path.append(path[-1] + delta) 

		if la.norm(f) < 0.001: break

	return path


def min_mode():
	
	# Initial random move
	norm  = 2*(np.random.rand(2)-0.5)
	norm /= la.norm(norm)
	path  = [np.array(norm)/10]

	for n in range(0,100):
		
		h = pots.hess(path[-1])
		g = pots.grad(path[-1])

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
			delta -= evecs[i] * np.dot(g, evecs[i]) * sign / evals[i]

		# Impose max step size
		if la.norm(delta) > 0.1:
			delta = 0.1 * delta / la.norm(delta)

		path.append(path[-1] + delta)
		if la.norm(g) < 0.01:
			break

	return path

def plot_path(path, show_min_evec=False):

	plt.plot(*zip(*path), marker="+")

	if show_min_evec:
		for p in path:
			h = pots.hess(p)
			evals, evecs = la.eig(h)
			imin = list(evals).index(min(evals))
			plt.plot(*zip(p, p+evecs[imin]/10))

def plot_forces(path):
	plt.plot([la.norm(pots.grad(p)) for p in path])

paths = []
for n in range(0,100):
	try:
		paths.append(min_mode())
	except:
		continue

plt.subplot(121)
pots.plot()
for p in paths: plot_path(p)

plt.subplot(222)
plt.xlabel("Iteration")
plt.ylabel("|Force|")
for p in paths: plot_forces(p)

plt.show()
