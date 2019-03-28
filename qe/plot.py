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

		label = "none"
		for l in lines:
			if "!" in l:
				energy = float(l.split("=")[-1].split("R")[0])
			if "Total force" in l:
				force = float(l.split()[3])
			if "P=" in l:
				pressure = float(l.split("=")[-1])
			if "unit-cell volume" in l:
				volume = float(l.split("=")[-1].split("(")[0])
			if "LABEL:" in l:
				label = l.split(":")[-1].lower()

		data.append([iteration, energy, force, pressure, volume, label])

	if len(data) != 0:
		data.sort()
		ns, es, fs, ps, vs, ls = zip(*data)

		colors = {
			"none":"blue",
			"relax":"green",
			"saddle_point_search":"red"
		}

		for i in range(0, len(data)-1):
			
			col = colors[ls[i]]

			plt.subplot(221)
			plt.plot(ns[i:i+2], es[i:i+2], color=col)
			plt.xlabel("Iteration")
			plt.ylabel("Energy")

			plt.subplot(222)
			plt.plot(ns[i:i+2], fs[i:i+2], color=col)
			plt.axhline(0)
			plt.xlabel("Iteration")
			plt.ylabel("|Force| (Ry/au)")

			plt.subplot(223)
			plt.plot(ns[i:i+2], ps[i:i+2], color=col)
			plt.axhline(0)
			plt.xlabel("Iteration")
			plt.ylabel("Pressure (kbar)")

			plt.subplot(224)
			plt.plot(ns[i:i+2], vs[i:i+2], color=col)
			plt.xlabel("Iteration")
			plt.ylabel("Volume (a.u^3)")

	plt.draw()
	plt.pause(0.5)
	plt.clf()
