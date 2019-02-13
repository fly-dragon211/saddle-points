import sys
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

steps = []
step = {}
lines = open(sys.argv[1]).read().split("\n")
for i, line in enumerate(lines):
	line = line.strip()

	if len(line) == 0: 
		if len(step) > 0: steps.append(step)
		step = {}
		continue
	
	name, vals = line.split(":")
	vals = vals.split(",")
	try:
		step[name] = np.array([float(v) for v in vals])
	except:
		step[name] = vals

nx, ny = [3,3]
plot_count = 1

for p in range(0,len(steps[0]["Value"])-1):
	
	plt.subplot(ny, nx, plot_count)
	plot_count += 1
	plt.plot([steps[0]["Value"][p]],[steps[0]["Value"][p+1]],marker=".",markersize=20)
	#xs = [s["Value"][p] for s in steps]
	#ys = [s["Value"][p+1] for s in steps]
	#plt.plot(xs,ys, color="blue", label="Path")
	plt.xlabel(steps[0]["Name"][p])
	plt.ylabel(steps[0]["Name"][p+1])
	plt.plot(steps[0]["Value"][p],steps[0]["Value"][p+1],color="red", label="Activation")
	plt.plot(steps[0]["Value"][p],steps[0]["Value"][p+1],color="green", label="Relaxation")
	plt.plot(steps[0]["Value"][p],steps[0]["Value"][p+1],color="black", label="Force")
	plt.plot(steps[0]["Value"][p],steps[0]["Value"][p+1],color="grey", label="Normal")
	plt.legend(loc="best")

	for s in steps:
		x = s["Value"][p]
		y = s["Value"][p+1]
		xact = s["Activation"][p]
		yact = s["Activation"][p+1]
		xmin = s["Relaxation"][p]
		ymin = s["Relaxation"][p+1]
		xnorm = s["Normal"][p]
		ynorm = s["Normal"][p+1]
		xf = s["Force"][p]
		yf = s["Force"][p+1]
		
		xs = xact+xmin
		ys = yact+ymin
		step_length = np.sqrt(xs*xs+ys*ys)

		fnorm = np.sqrt(xf*xf+yf*yf)
		xf *= step_length/(10*fnorm)
		yf *= step_length/(10*fnorm)

		xnorm *= step_length/10
		ynorm *= step_length/10

		act_color = "red"
		relax_color = "red"
		if "Minimization" in s["Path"]:
			act_color = "purple"
			relax_color = "blue"

		plt.plot([x,x+xact], [y,y+yact], color="red", linestyle="--")
		plt.plot([x+xact, x+xact+xmin], [y+yact, y+yact+ymin], color="green", linestyle="--")
		plt.plot([x,x+xnorm], [y,y+ynorm], color="grey")
		plt.plot([x,x+xf], [y,y+yf], color="black")

plt.subplot(ny, nx, plot_count)
plot_count += 1
plt.plot([s["Potential"][0] for s in steps])
plt.xlabel("Iteration")
plt.ylabel("Potential")
plt.legend(loc="best")

plt.subplot(ny, nx, plot_count)
plot_count += 1
plt.plot([la.norm(s["Force"]) for s in steps])
plt.axhline(0,color="black")
plt.xlabel("Iteration")
plt.ylabel("|Force|")

if len(steps) > 1:
	plt.subplot(ny, nx, plot_count)
	plot_count += 1
	deltas = [np.array(steps[i]["Value"])-np.array(steps[i-1]["Value"]) for i in range(1,len(steps))]
	plt.plot([la.norm(d) for d in deltas])
	plt.ylim([0, max([la.norm(d)*1.1 for d in deltas])])
	plt.xlabel("Iteration")
	plt.ylabel("Step size")

plt.subplot(ny, nx, plot_count)
plot_count += 1
disp = [0]
for i in range(1,len(steps)):
	disp.append(disp[-1]+la.norm(steps[i]["Value"]-steps[i-1]["Value"]))
plt.plot(disp, [s["Potential"][0] for s in steps])
plt.xlabel("Displacement along path")
plt.ylabel("Potential")

plt.show()
