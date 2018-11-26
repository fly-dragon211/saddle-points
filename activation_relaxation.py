import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

EPS = 0.01
pot_evals = 0

def len_jones(x):
	if len(x) % 3 != 0:
		print "Error: lenard jones tried to operate on an incorrect dimension object!"
		quit()

	ps = []
	for i in range(0,len(x)/3):
		ps.append(np.array(x[i*3:i*3+3]))

	ret = 0
	for i in range(0,len(ps)):
		for j in range(i+1,len(ps)):
			r = np.linalg.norm(ps[i]-ps[j])
			ret += 1/(r**12) - 2/(r**6)
	return ret


def plot_len_jones():
	xy = []
	for x in np.arange(0.8,5,0.01):
		xy.append([x,len_jones([0,0,0,x,0,0])])	
	plt.plot(*zip(*xy))
	plt.show()


def eggbox(x,y):
	return np.sin((x-0.5)*np.pi)*np.cos(y*np.pi)

def eggbox_unbound(x,y):
	return eggbox(x,y)*np.exp(-8*(x*x+y*y))

def eggbox_bound(x,y):
	return eggbox(x,y)*(x**2+y**2)

def square_lj(x,y):
	pos = [0,0,0,1,0,0,-1,0,0,0,1,0,0,-1,0]
	pos[0] += x
	pos[1] += y
	return len_jones(pos)

def potential_xy(x,y):
	global pot_evals
	pot_evals += 1

	return square_lj(x,y)
	return eggbox(x,y)
	return eggbox_bound(x,y)
	return eggbox_unbound(x,y)

def potential(x):
	return potential_xy(x[0],x[1])

def hessian(x):
	global EPS
	ret = np.zeros((len(x), len(x)))
	for i in range(0,len(x)):
		for j in range(0, len(x)):
			ip = np.zeros(len(x))
			jp = np.zeros(len(x))
			ip[i] = EPS
			jp[i] = EPS
			hij  = potential(x+ip+jp)
			hij -= potential(x+ip-jp)
			hij -= potential(x-ip+jp)
			hij += potential(x-ip-jp)
			hij /= 4*EPS*EPS
			ret[i][j] = hij
	return ret

def force(x):
	global EPS
	x = np.array(x)
	ret = np.zeros(len(x))
	for i in range(0,len(x)):
		ip      = np.zeros(len(x))
		ip[i]   = EPS/2
		ret[i]  = potential(x-ip)-potential(x+ip)
		ret[i] /= EPS
	return ret

def plot_potential(log_scale = False):
	RANGE = 2.5
	
	x = np.linspace(-RANGE,RANGE,100)
	y = np.linspace(-RANGE,RANGE,100)
	z = []
	for xi in x:
		row = []
		for yi in y:
			p = potential_xy(xi,yi)
			if p > 100:
				log_scale = True
			row.append(p)
		z.append(row)
	if log_scale:
		z = np.sign(z)*np.log(np.abs(z))
	x,y = np.meshgrid(x,y)
	plt.contour(x,y,z,cmap="seismic",alpha=0.2)
	im = plt.imshow(z,extent=(-RANGE,RANGE,-RANGE,RANGE), origin="lower",cmap="seismic",alpha=0.1, interpolation="bilinear")
	cbar = plt.colorbar(im,orientation="horizontal",shrink=0.8)
	cbar.set_label("Potential")


def find_saddle_art(
	init_rand_dist  = 0.1,
	step_scale      = 0.2,
	step_tol        = 0.02,
	par_force_scale = 1,
	per_force_scale = 1,
	plot            = False,
	local_min = np.array([0,0])
	):

	global pot_evals
	pot_evals = 0
	x = local_min + 2*0.1*(np.random.rand(len(local_min))-0.5)
	path = [local_min, x]
	prev_dx = x-local_min

	for n in range(1,100):

		xh   = x - local_min
		xh  /= np.linalg.norm(xh)
		f    = force(x)
		f    = np.tanh(np.linalg.norm(f))*f/np.linalg.norm(f)
		fpar = np.matmul(f,xh)*xh
		fper = f - fpar 
		g    = per_force_scale*fper - par_force_scale*fpar
		dx   = g*step_scale
		x   = x + dx 
		path.append(x)

		if np.dot(prev_dx/np.linalg.norm(prev_dx), dx/np.linalg.norm(dx)) < -0.1:
			step_scale /= 2
			step_tol   /= 2

		prev_dx = dx
		if np.linalg.norm(dx) < step_tol:
			break

	if plot:
		# Plot the path taken through config space
		potential_path = []
		for p in path:
			potential_path.append(potential(p))
		xs, ys = zip(*path)
		plt.plot(xs,ys,linewidth=2,marker=".")
		plt.xlabel("x")
		plt.ylabel("y")

	return [pot_evals, len(path)]

def par_opt():
	pe = []
	for pfs in np.arange(0.5,1.5,0.01):
		pes = []
		for i in range(0,500):
			find_saddle_art(per_force_scale=pfs)
			pes.append(pot_evals)
		pe.append([pfs, np.mean(pes), np.std(pes)])

	x,y,dy = np.array(zip(*pe))
	plt.plot(x,y)
	plt.fill_between(x,y+dy/2,y-dy/2,alpha=0.2)
	plt.xlabel("Perpendicular force scale")
	plt.ylabel("Average potential evaluations")

def plot_several(n):
	dat = []
	for i in range(0,n):
		dat.append(find_saddle_art(plot=True))
	pevs, lens = zip(*dat)
	print "Average potential evals:",np.mean(pevs),"Average path length:",np.mean(lens)
	plot_potential()

#par_opt()
plot_several(10)
#plot_potential()
plt.show()
