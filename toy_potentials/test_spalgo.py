import spalgo
import numpy as np
import numpy.linalg as la
import potentials as pots
import matplotlib.pyplot as plt

def test_method(method):

	pots.plot()
	ITER = 10
	evals = []
	successes = 0
	for n in range(0,ITER):

		pots.pot_evals = 0
		path = method(pots.pot, pots.grad, pots.hess)
		succ = True if la.norm(pots.grad(path[-1])) < 0.1 else False
		if succ: successes += 1
		evals.append(pots.pot_evals)
		plt.plot(*zip(*path), marker="+")

	plt.xlabel(str(method)+
	"\nPotential: "+str(pots.current)+
	"\nPot evals: "+str(np.mean(evals))+
	" +/- "+str(np.std(evals))+
	"\nSuccesses: "+str(successes)+"/"+str(ITER))

n_pots  = len(pots.avail_pots)
n_meths = len(spalgo.methods) 
for i, pot in enumerate(pots.avail_pots):
	for j, meth in enumerate(spalgo.methods):
		pots.current = pot
		plt.subplot(n_pots,n_meths,1+i+j*n_pots)
		test_method(meth)

plt.show()
