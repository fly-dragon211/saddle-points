import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import potentials as pots
from scipy.optimize import curve_fit

def steepest_decent(plot=False):
	x    = np.random.rand(2)
	path = [np.array(x)]
	while True:
		g = pots.gradient(x)
		w = 0.1
		x = x - w*g 
		path.append(np.array(x))
		if la.norm(g) < 0.1:
			break
	if plot:
		plt.plot(*zip(*path), marker=".")

def mode_climb_force(x, xmin):

	evals, evecs = la.eig(pots.hessian(x))
	pairs = zip(evals, evecs)
	pairs.sort()
	l1, v1 = pairs[0]

	force = -pots.gradient(x)
	if l1 > 0:
		feff = -np.dot(force,v1)*v1
	else:
		feff = force - 2*np.dot(force,v1)*v1
	return feff

def min_mode_climb(plot=False, dx2=0.01, dxm=0.1):

	xmin = np.zeros(2)
	x    = xmin	
	path = [np.array(x)] 

	# Move along random direction until -ve eigenvale encountered
	rand = (2*np.random.rand(len(xmin))-1)
	for n in range(0,10):
		x += rand*0.1
		evals, evecs = la.eig(pots.hessian(x))
		pairs = zip(evals, evecs)
		l1, v1 = pairs[0]
		path.append(np.array(x))
		if l1 < 0:
			break

	# Move along minimum mode
	for n in range(0,100):
		
		f1   = mode_climb_force(x, xmin)
		if la.norm(f1) < 0.0001:
			break

		f1hat = f1/la.norm(f1)

		x2    = x + dx2*f1hat
		f2    = mode_climb_force(x2, xmin)
		f2hat = f2/la.norm(f2)
		
		fmidway = np.dot(f2+f1,f1hat)/2
		cmidway = np.dot(f2-f1,f1hat)/dx2
		dx = -fmidway/cmidway + dx2/2

		if abs(dx) > dxm:
			dx = np.sign(dx)*dxm

		x += dx*f1hat
		path.append(np.array(x))

	if plot:
		plt.plot(*zip(*path), marker=".")

def min_mode_linesearch(plot=False):
	xmin = np.zeros(2)
	x = np.array(xmin)
	path = [np.array(xmin)]
	line_searches = []
	dlast = np.zeros(2)
	for n in range(0,100):

		# Get direction along minimum mode that
		# is best aligned with the path so far
		evals, evecs = la.eig(pots.hessian(x))
		mini = list(evals).index(min(evals))
		dhat = evecs[mini]/la.norm(evecs[mini])
		if np.dot(dhat, dlast) < 0: # Pointing the wrong way
			dhat = -dhat
		dlast = dhat


		# Perform a line search along this direction
		# until the energy decreases
		xl = np.array(x)
		elast = pots.potential(xl)
		for m in range(0,10):
			xl += dhat * 0.01
			ehere = pots.potential(xl)
			line_searches.append(np.array(xl))
			de = ehere - elast
			if de > 0:
				x = xl
			else:
				break

		# Equilibriate perpedicular component of force
		# (using a line search until the energy increases)
		force = - pots.gradient(x)
		force_para = np.dot(force,dhat)*dhat
		force_perp = force - force_para
		fhat = force_perp/la.norm(force_perp)
		xl = np.array(x)
		elast = pots.potential(xl)
		for m in range(0,10):
			xl += fhat * 0.01
			ehere = pots.potential(xl)
			line_searches.append(np.array(xl))
			de = ehere - elast
			if de < 0:
				x = xl
			else:
				break

		path.append(np.array(x))

	if plot:
		path += np.random.rand(len(path),2)*0.01
		plt.plot(*zip(*path), marker="+",markersize=10)
		plt.scatter(*zip(*line_searches), marker=".", alpha=0.5)
		print len(path)

def min_mode_force_proj(axes, pot_axis=None):
	
	x = np.zeros(2)
	path = [np.array(x)]
	path_force = []
	path_pot = []
	feff_last = np.zeros(2)
	force_last = np.zeros(2)
	second_half = False

	max_steps = 10
	for n in range(0,max_steps):

		force          = - pots.gradient(x)
		eigval, eigvec = pots.min_mode(x) 
		fpara          = np.dot(force, eigvec)*eigvec

		if eigval >= 0:
			feff = -fpara
		else:
			feff = force - 1.1*fpara
		
		if len(axes) > 1:
			if n % (max_steps/10) == 0:
				linesearch = []
				for m in np.linspace(0,1,10):
					xl  = x + m*feff/la.norm(feff)
					val = pots.potential(xl)
					linesearch.append([m, val])
				xs,ys = zip(*linesearch)
				if pots.log_scale():
					ys = np.log10(np.abs(ys))
				r = n / float(max_steps)
				axes[1].plot(xs, ys, label=str(n/10), color=[r,0,1-r])
				axes[1].set_xlabel("Progress along linesearch")
				axes[1].set_ylabel("Potential")

		#if second_half:
		#	if la.norm(force) > la.norm(force_last):
		#		break
		#elif la.norm(force) < la.norm(force_last):
		#	second_half = True

		feff += feff_last
		x += (1/float(max_steps))*feff/la.norm(feff)
		path.append(np.array(x))

		path_force.append(la.norm(force))
		path_pot.append(pots.potential(x))

		feff_last  = feff
		force_last = force

	if len(axes) > 0:
		axes[0].plot(path_force)
		axes[0].set_xlabel("Iteration")
		axes[0].set_ylabel("|Force|")

	if len(axes) > 2:
		axes[2].plot(path_pot)
		axes[2].set_xlabel("Iteration")
		axes[2].set_ylabel("Potential")

	if pot_axis != None:
		lab  = "Points: "+str(len(path))
		pot_axis.plot(*zip(*path), marker=".",label=lab)
		pot_axis.legend()


def line_min_fit(x, pot, x0, sq):
	return pot + sq*(x-x0)*(x-x0)

def line_minimize(x_start, direction):

	step = 0.01
	d = direction/la.norm(direction)

	mapped_pos = [x_start]
	mapped_pot = [pots.potential(x_start)]
	mapped_del = [0]
	first = mapped_pot[0]

	while True:
		min_index = mapped_pot.index(min(mapped_pot))
		if min_index == 0:
			xtest = mapped_pos[0]-step*d
			mapped_pos.insert(0, xtest)
			mapped_pot.insert(0, pots.potential(xtest))
			mapped_del.insert(0, mapped_del[0]-step)
		elif min_index == len(mapped_pot)-1:
			xtest = mapped_pos[-1]+step*d
			mapped_pos.append(xtest)
			mapped_pot.append(pots.potential(xtest))
			mapped_del.append(mapped_del[-1]+step)
		else:
			break

	success = True
	try:
		par, covar = curve_fit(line_min_fit, mapped_del, mapped_pot, p0=[min(mapped_pot), 0, 1])
	except:
		success = False

	if success:
		return par[1]*d + x_start
	else:
		return mapped_pos[mapped_pot.index(min(mapped_pot))]

def min_mode_plane_min(axes=[], pot_axis=None):

	rand_dist = 0.001
	x = np.zeros(2)
	pot_scale = pots.potential(x)
	x += rand_dist*(2*np.random.rand(2)-1)
	pot_scale = pots.potential(x) - pot_scale
	pot_scale /= rand_dist
	if pot_scale < 0:
		print "Error, random inital move move reduced energy!"
		quit()

	path = [np.array(x)]
	path_pot = []
	path_dp  = []
	path_force = []
	steps_res = 10
	pot_last = pots.potential(x)

	for n in range(0,steps_res*2):

		force          = -pots.gradient(x)
		eigval, eigvec = pots.min_mode(x) 
		fpara          = np.dot(force, eigvec)*eigvec
		fperp          = force - fpara

		if eigval >= 0:
			feff = -fpara
		else:
			feff = force - 2*fpara

		x += feff/(float(steps_res)*la.norm(feff))
		x  = line_minimize(x, fperp)

		if len(path_pot) > 1:
			dp = path_pot[-1] - path_pot[-2]
			path_dp.append(dp)
			if dp < 0:
				break

		path.append(np.array(x))
		path_pot.append(pots.potential(x))
		path_force.append(la.norm(feff))
	
	if len(axes) > 0:
		axes[0].plot(path_pot)
		axes[0].set_xlabel("iteration")
		axes[0].set_ylabel("potential")

	if len(axes) > 1:
		axes[1].plot(path_force)
		axes[1].set_xlabel("iteration")
		axes[1].set_ylabel("|force|")

	if len(axes) > 2:
		axes[2].plot(path_dp)

	if pot_axis != None:
		lab = "Points: " + str(len(path))
		pot_axis.plot(*zip(*path),label=lab)
	

def test_method(method, repeats):

	ps = pots.all_potentials
	plots = 4
	for npot in range(0,len(ps)):
		pots.current_potential = ps[npot]
		axes = [plt.subplot(len(ps), plots, plots*npot+i+1) for i in range(0,plots-1)]
		for testn in range(0,repeats):
			method(axes, pot_axis=plt.subplot(len(ps), plots, plots*npot+plots))
		pots.plot_potential()

test_method(min_mode_plane_min, 1)
plt.show()
