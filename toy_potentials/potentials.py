import numpy as np
import matplotlib.pyplot as plt

def eggbox(x):
	x0 = (x[0]-0.5)*np.pi
	x1 = x[1]*np.pi
	return np.sin(x0)*np.cos(x1)# + 0.1*np.sin(x0+x1)

pot_evals = 0
def pot(x):
	global pot_evals
	pot_evals += 1
	return eggbox(x)

def grad_fd(x, f=pot):
	EPS = 0.01
	p0 = f(x)
	px = f([x[0]+EPS, x[1]])
	py = f([x[0], x[1]+EPS])
	return np.array([px-p0,py-p0])/EPS

def hess_fd(x, f=pot):
	EPS = 0.01
	hess = np.identity(len(x), float)
	for i in range(0,len(x)):
		di = np.zeros(len(x))
		di[i] = EPS
		for j in range(0,len(x)):
			dj = np.zeros(len(x))
			dj[j] = EPS
			hess[i][j] = f(x+di+dj)-f(x+di-dj)-f(x+dj-di)+f(x-di-dj)
	hess = 0.5 * (hess + hess.T)
	return hess / (4*EPS**2)

def grad(x):
	return grad_fd(x)

def hess(x):
	return hess_fd(x)

def plot(to_plot=pot):
	x = np.linspace(-1.5,1.5,100)
	y = np.linspace(-1.5,1.5,100)
	x,y = np.meshgrid(x,y)
	z = []
	for i in range(0,len(x)):
		row = []
		for j in range(0,len(x[i])):
			val = to_plot([x[i][j],y[i][j]])
			row.append(val)
		z.append(row)
	cs = plt.contour(x,y,z)
	plt.clabel(cs, inline=1, fontsize=10)
