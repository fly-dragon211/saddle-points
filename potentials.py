import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

all_potentials = []
pot_evals = 0
track_pot_evals = False
def potential(x):
	global current_potential, pot_evals
	if track_pot_evals:
		pot_evals += 1
	return current_potential(x)

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
	
def pot_eggbox(x):
        return np.sin((x[0]-0.5)*np.pi)*np.cos(x[1]*np.pi)
all_potentials.append(pot_eggbox)

def pot_eggbox_twisted(x):
	t    = la.norm(x)
	c, s = np.cos(t), np.sin(t)
	r    = np.array([[c,-s],[s,c]])
	return pot_eggbox(np.matmul(r,x))
all_potentials.append(pot_eggbox_twisted)

def pot_square_lj(x):
        pos = [0,0,0,1,0,0,-1,0,0,0,1,0,0,-1,0]
        pos[0] += x[0]
        pos[1] += x[1]
        return len_jones(pos)
all_potentials.append(pot_square_lj)

def pot_square_lj_twisted(x):
	t    = la.norm(x)
	c, s = np.cos(t), np.sin(t)
	r    = np.array([[c,-s],[s,c]])
	return pot_square_lj(np.matmul(r,x))
all_potentials.append(pot_square_lj_twisted)

def plot_potential():
	RANGE = 2
	RESOL = 50
	x   = np.linspace(-RANGE,RANGE,RESOL)
	y   = np.linspace(-RANGE,RANGE,RESOL)
	z   = []
	all_vals = []
	for yi in y:
		row = []
		for xi in x:
			p = np.array([xi,yi])
			val = potential(p)
			row.append(val)
			all_vals.append(val)
		z.append(row)
	x,y = np.meshgrid(x,y)
	if max(all_vals) > 100:
		z = np.log(np.abs(z)) * np.sign(z)
	plt.contour(x,y,z,cmap="seismic")
	plt.imshow(z,cmap="seismic",extent=(-RANGE,RANGE,-RANGE,RANGE),origin="lower",interpolation="bilinear",alpha=0.2)

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
        return 0.5*(ret + np.transpose(ret))

def min_mode(x):
	evals, evecs = la.eig(hessian(x))
	mini = list(evals).index(min(evals))
	return [evals[mini], evecs[mini]]

def gradient(x):
        global EPS
        x   = np.array(x)
	p0  = potential(x)
        ret = np.zeros(len(x))
        for i in range(0,len(x)):
                ip      = np.zeros(len(x))
                ip[i]   = EPS
                ret[i]  = potential(x+ip) - p0
                ret[i] /= EPS
        return ret

def log_scale():
	if current_potential in [pot_square_lj, pot_square_lj_twisted]:
		return True
	return False
	
EPS = 0.01
current_potential = pot_eggbox
