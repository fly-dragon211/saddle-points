from cell import cell
import numpy as np
import sys
import os

os.system("rm -r singlepoints")
c = cell(sys.argv[1])

if False:
	# Perturb the first atom a little (for testing)
	disp = []
	for i in range(0, len(c.atoms)):
		disp.append(np.zeros(3))
	disp[0] = [0.1,0,0]
	c.perturb_atoms_cart(disp)

for i in range(0,100):

	c.run_singlepoint_qe(sys.argv[1])
	
	# Due to translational invariance, we may
	# keep the first atom fixed
	pert = c.forces_cart
	pert[0] = [0,0,0]

	# Simply follow the force downhill
	c.perturb_atoms_cart(pert)
	c.apply_strain_cart(c.stress_cart/1000)

	if c.total_force  > 10e-8: continue
	if c.total_stress > 0.01: continue
	break
