import sys
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

path = []
path_pot = []
path_force = []

for line in open(sys.argv[1]).read().split("\n"):
	try:
		xstr, pot_str, force_str = line.split(":")
	except:
		continue

	path.append(np.array([float(x) for x in xstr.split()]))
	path_pot.append(float(pot_str))
	path_force.append(np.array([float(f) for f in force_str.split()]))

px = 2
py = 2

plt.subplot(py, px, 1)
plt.plot(path_pot)
plt.xlabel("Iteration")
plt.ylabel("Potential")

plt.subplot(py, px, 2)
plt.plot([la.norm(f) for f in path_force])
plt.xlabel("Iteration")
plt.ylabel("|Force|")

plt.subplot(py, px, 3)
plt.plot([la.norm(x-path[0]) for x in path])
plt.xlabel("Iteration")
plt.ylabel("|x|")

plt.show()
