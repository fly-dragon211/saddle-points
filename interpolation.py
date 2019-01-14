import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

EPS = 0.0001

def interp_func(x):
        x = np.array(x)
        return x*x*x*x
        #return x**4
        return np.cos(x*np.pi)

def grad_interp_func(x):
        global EPS
        return (interp_func(x+EPS)-interp_func(x))/EPS

def interp(x, x1, x2, f1, f2, g1, g2):
        m = np.array([
        [1,x1,x1**2,x1**3],
        [1,x2,x2**2,x2**3],
        [0,1,2*x1,3*x1**2],
        [0,1,2*x2,3*x2**2]
        ])
        try:
                minv = la.inv(m)
        except:
                print "Singular matrix!"
                print m
                quit()
        b = [f1,f2,g1,g2]
        a = np.matmul(minv,b)
        return a[0] + a[1]*x + a[2]*x**2 + a[3]*x**3

def interp_new(x, xs, fs):
        
        M = len(xs)
 
        m = np.zeros((4*(M-1), 4*(M-1)))
        b = np.zeros(4*(M-1))
 
        for i in range(0, M-1):
 
                n  = i*4

                # Left (1) and right(2) boundaries of section
                x1 = xs[i]
                x2 = xs[i+1]
                f1 = fs[i]
                f2 = fs[i+1]
                # Passes through x1, f1 and x2, f2
                b[n] = f1
                b[n+1] = f2
                for p in range(0,4):
                        m[n][n+p] = x1**(p+0.0)
                        m[n+1][n+p] = x2**(p+0.0)

                # Boundary between sections
                n  = i*4
                nl = n-4
                nr = n+4

                if nl >= 0:
                        # Continuous first derivative at left boundary
                        b[n+2] = 0

                        m[n+2][n] = 0
                        m[n+2][n+1] = 1
                        m[n+2][n+2] = 2*x1
                        m[n+2][n+3] = 3*x1**2

                        m[n+2][nl] = 0
                        m[n+2][nl+1] = -1
                        m[n+2][nl+2] = -2*x1
                        m[n+2][nl+3] = -3*x1**2

                        # Continous second derivative at left boundary
                        b[n+3] = 0

                        m[n+3][n] = 0
                        m[n+3][n+1] = 0
                        m[n+3][n+2] = 1
                        m[n+3][n+3] = 3*x1

                        m[n+3][nl] = 0
                        m[n+3][nl+1] = 0
                        m[n+3][nl+2] = -1
                        m[n+3][nl+3] = -3*x1

        # Set d^3/dx^3 = 0 at far left/right edge
        # goes in rows 3 and 4 (indicies 2 and 3)
        # as the leftmost section has no left boundary

        b[2] = 0
        m[2][3] = 1
        b[3] = 0
        m[3][(M-2)*4+3] = 1

        minv = la.inv(m)
        coeff = np.matmul(minv, b)

        right_index = None
        for i in range(0, len(xs)):
                if x < xs[i]:
                        right_index = i
                        break

        if right_index == 0:
                print "Error x requested out of interpolation range! (too small)"
                print x,"<", min(xs)
                quit()
        if right_index == None:
                print "Error x requested out of interpolation range! (too large)"
                print x,">", max(xs)
                quit()

        ci = right_index-1
        c = coeff[ci*4:(ci+1)*4]
        return c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3


def test_interp_scheme():
        interp_points = np.linspace(-1.9,1.9,100)

        pot_evals = 6
        samp = np.linspace(-2,2,pot_evals)

        vals = interp_func(samp)
        interp_vals = []
        for p in interp_points:
                interp_vals.append(interp_new(p, samp, vals))

        plt.subplot(211)
        for x in samp: plt.plot([x],[interp_func(x)],marker="+")
        plt.plot(interp_points, interp_vals, linestyle = "--")
        plt.plot(interp_points, interp_func(interp_points))

        samp = np.linspace(-2,2,pot_evals/2)
        plt.subplot(212)
        for x in samp: plt.plot([x],[interp_func(x)],marker="+")
        for i in range(1,len(samp)):

                x1 = samp[i-1]
                x2 = samp[i]
                f1 = interp_func(x1)
                f2 = interp_func(x2)
                g1 = grad_interp_func(x1)
                g2 = grad_interp_func(x2)

                p = []
                xp = np.linspace(x1,x2,100)
                for x in xp: p.append(interp(x,x1,x2,f1,f2,g1,g2))
                plt.plot(xp, p, linestyle="--")

        plt.plot(interp_points, interp_func(interp_points))
        plt.show()

test_interp_scheme()
