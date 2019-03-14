import os
import matplotlib.pyplot as plt

data = []
for f in os.listdir("singlepoints"):
	if not f.endswith(".out"): continue

	iteration = int(f.split("_")[-1].split(".")[0])
	f = "singlepoints/"+f
	lines = open(f).read().split("\n")

	for l in lines:
		if "!" in l:
			energy = float(l.split("=")[-1].split("R")[0])

	data.append([iteration, energy])

data.sort()
ns, es = zip(*data)

plt.plot(ns, es)
plt.show()
