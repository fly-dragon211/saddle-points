import os
import matplotlib.pyplot as plt

while True:

	data = []
	for f in os.listdir("singlepoints"):
		if not f.endswith(".out"): continue

		try:
			iteration = int(f.split("_")[-1].split(".")[0])
		except:
			continue

		with open("singlepoints/"+f) as outfile:
			contents = outfile.read()
			if not "JOB DONE" in contents: continue
			lines = contents.split("\n")

		for l in lines:
			if "!" in l:
				energy = float(l.split("=")[-1].split("R")[0])
			if "Total force" in l:
				force = float(l.split()[3])
			if "P=" in l:
				pressure = float(l.split("=")[-1])
			if "unit-cell volume" in l:
				volume = float(l.split("=")[-1].split("(")[0])
		
		data.append([iteration, energy, force, pressure, volume])

	if len(data) != 0:
		data.sort()
		ns, es, fs, ps, vs = zip(*data)

		plt.subplot(221)
		plt.plot(ns, es)
		plt.xlabel("Iteration")
		plt.ylabel("Energy")

		plt.subplot(222)
		plt.plot(ns, fs)
		plt.axhline(0)
		plt.xlabel("Iteration")
		plt.ylabel("|Force| (Ry/au)")

		plt.subplot(223)
		plt.plot(ns, ps)
		plt.axhline(0)
		plt.xlabel("Iteration")
		plt.ylabel("Pressure (kbar)")

		plt.subplot(224)
		plt.plot(ns, vs)
		plt.xlabel("Iteration")
		plt.ylabel("Volume (a.u^3)")

		plt.draw()
	plt.pause(0.5)
	plt.clf()
