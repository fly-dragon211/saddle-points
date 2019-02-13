import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.optimize import minimize

pot_evals = 0
def pot(x):
	global pot_evals
	pot_evals += 1

	x = np.pi*np.array(x)

	r = 4*la.norm(x)/10
	c = np.cos(r)
	s = np.sin(r)
	rot = np.array([[c,-s],[s,c]])
	x = np.matmul(rot,x)

	return -np.cos(x[0])*np.cos(x[1])

def grad(x):

	eps = 0.001
	ret = np.zeros(len(x))
	for i in range(0, len(x)):
		ei    = np.zeros(len(x))
		ei[i] = eps
		ret[i] = (pot(x+ei)-pot(x))/eps
	return ret

	x = np.pi*np.array(x)
	return np.pi*np.array([np.sin(x[0])*np.cos(x[1]),
			       np.cos(x[0])*np.sin(x[1])])

def hess(x):

	eps = 0.001
        ret = np.zeros((len(x), len(x)))
        for i in range(0,len(x)):
                for j in range(0, len(x)):
                        ip = np.zeros(len(x))
                        jp = np.zeros(len(x))
                        ip[i] = eps
                        jp[j] = eps
                        hij  = pot(x+ip+jp)
                        hij -= pot(x+ip-jp)
                        hij -= pot(x-ip+jp)
                        hij += pot(x-ip-jp)
                        hij /= 4*eps*eps
                        ret[i][j] = hij

        return 0.5*(ret + ret.T)

	x = np.pi*np.array(x)
	xy = -np.pi**2*np.sin(x[0])*np.sin(x[1])
	xx =  np.pi**2*np.cos(x[0])*np.cos(x[1])
	return np.array([[xx,xy],[xy,xx]])

def min_hess_eval(x): return min(la.eig(hess(x))[0])
def neg_hess_evals(x): return sum([1 for e in la.eig(hess(x))[0] if e < 0])
def hess_det(x): return la.det(hess(x))

def plot_xy_func(xy_func):
	xs = np.linspace(-2,2,100)
	ys = np.linspace(-2,2,100)
	extent = (-2,2,-2,2)
	z = []
	for y in ys:
		row = []
		for x in xs:
			row.append(xy_func([x,y]))
		z.append(row)

	xs,ys = np.meshgrid(xs,ys)
	plt.contour(xs,ys,z,alpha=0.5)
	plt.imshow(z, extent=extent, interpolation="bilinear", origin="lower", alpha=0.3)

def path_to_local_min(func, x0, grad_tol=0.01):

	path = [x0]
	for n in range(0,100):
		min_res = minimize(func, path[-1], method="BFGS", options={"maxiter":1})
		path.append(min_res.x)
		if la.norm(min_res.jac) < grad_tol:
			break
	return path

def path_to_saddle():
	
	x = 0.01*(np.random.rand(2)*2-1)
	path = [x.copy()]
	pot_last = pot(x)

	for iteration in range(0,100):

		f     = -grad(x)
		p     = pot(x)

		if p < pot_last:
			normf = lambda x : la.norm(grad(x))
			min_res = minimize(normf, x,
					   options={"gtol":0.01})
			path.append(min_res.x)
			break

		fhat  = f/la.norm(f)

		norm  = x.copy()
		norm /= la.norm(norm)

		fpara = np.dot(fhat, norm)*norm
		fperp = fhat - fpara 

		x += 0.1 * (fperp/la.norm(fperp) - fpara/la.norm(fpara))

		cons = lambda p : np.dot(x-p, fpara)
		min_res = minimize(pot, x, constraints = {"type":"eq", "fun":cons },
				   options={"gtol":0.01})

		disp = min_res.x - x
		if la.norm(disp) > 0.1: disp = 0.1*disp/la.norm(disp)
		x += disp

		path.append(x.copy())
		pot_last = p

	return path

plot_xy_func(pot)
for n in range(0,10):
	pot_evals = 0
	path = path_to_saddle()
	print pot_evals
	plt.plot(*zip(*path), marker="+")

plt.show()
