import spalgo
import potentials as pots
import matplotlib.pyplot as plt

ITER = 100

plt.subplot(221)
pots.plot()
tot = 0
for n in range(0,ITER):
	pots.pot_evals = 0
	path = spalgo.min_mode(pots.grad, pots.hess)
	tot += pots.pot_evals
	plt.plot(*zip(*path), marker="+")
plt.xlabel("Mode following\naverage pot evals: "+str(tot/ITER))

plt.subplot(222)
pots.plot()
tot = 0
for n in range(0,ITER):
	pots.pot_evals = 0
	path = spalgo.act_relax(pots.grad)
	tot += pots.pot_evals
	plt.plot(*zip(*path), marker="+")
plt.xlabel("Activation relaxation\naverage pot evals: "+str(tot/ITER))

plt.show()
