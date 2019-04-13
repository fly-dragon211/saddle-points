import pyKriging  
import potentials as pots
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import spalgo

def func(x):
	ret = []
	for xi in x:
		ret.append(pots.pot(xi))
	return np.array(ret)

def test1():

	sp = samplingplan(2)  
	x = []
	for n in range(0,4):
		x.append(np.random.rand(2)*2-1)
	x = np.array(x) * 2

	# Next, we define the problem we would like to solve
	testfun = func
	y = testfun(x)

	# Now that we have our initial data, we can create an instance of a Kriging model
	k = kriging(x, y, testfunction=testfun, name='simple')  
	k.train()

	# Now, five infill points are added. Note that the model is re-trained after each point is added
	numiter = 10
	for i in range(numiter):  
	    print 'Infill iteration {0} of {1}....'.format(i + 1, numiter)
	    newpoints = k.infill(1)
	    for point in newpoints:
		k.addPoint(point, testfun([point])[0])
	    k.train()

	# And plot the results
	k.plot()

def min_mode():

        # Initial random move
        norm  = 2*(np.random.rand(2)-0.5)
        norm /= la.norm(norm)
        path  = [np.array(norm)/10]

	# Start krig model with origin and random displacement
	krig_init = np.array([[0,0], path[-1]])
	k = kriging(krig_init, func(krig_init), testfunction=func, name='simple')  
	k.train()
	plt.show()

        for n in range(0,100):

		v = pots.pot(path[-1])
		g = pots.grad(path[-1])

		h = pots.hess_fd(path[-1], k.predict)

		k.addPoint(path[-1], v)
		k.train()

		plt.subplot(221)
		pots.plot(k.predict)
		plt.plot(*zip(*path))
		
		plt.subplot(222)
		pots.plot()
		plt.plot(*zip(*path))

		plt.draw()
		plt.pause(0.001)
		plt.clf()

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

        return path

def krig_sp():
	
        # Initial random move
        norm  = 2*(np.random.rand(2)-0.5)
        norm /= la.norm(norm)
        path  = [np.array(norm)/10]

	# Start krig model with origin and random displacement
	sp = samplingplan(len(norm)) 
	hc = 2*(sp.optimallhc(2**len(norm))-0.5)
	krig_init = [[0.0,0.0]]
	for p in hc: krig_init.append(p)
	krig_init = np.array(krig_init)
	k = kriging(krig_init, func(krig_init), testfunction=func, name='simple')  
	k.train()
	k.plot()

	last_found = norm
	for n in range(0,100):

		plt.subplot(221)
		pots.plot(k.predict)

		found   = []
		weights = []
		for n in range(0,10):
			path = spalgo.act_relax_fd(k.predict,
				max_range=1.2,
				max_step=0.01,
				max_iter=1000)
			plt.plot(*zip(*path))
			pt = np.array(path[-1])

			duplicate = False
			for i, p in enumerate(found):
				if la.norm(pt - p) < 0.1:
					weights[i] += 1
					duplicate = True
					break
			if duplicate:
				continue

			found.append(pt)
			weights.append(1)

		imax = weights.index(max(weights))
		last_found = found[imax]
		plt.plot([last_found[0]], [last_found[1]], marker="+")

		plt.subplot(222)
		pots.plot(pots.pot)
		plt.plot(*zip(*path))

		k.addPoint(path[-1], func([last_found]))
		k.train()

		plt.draw()
		plt.pause(0.1)
		plt.clf()

test1()
#krig_sp()
