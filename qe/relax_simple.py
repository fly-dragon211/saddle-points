from cell import cell, last_cell, clear_calc
import numpy as np
import sys
import os

if len(sys.argv) < 2:
	# Load the cell from the last singlepoint
	c = last_cell()
else:
	# Load the cell from the specified file
	# and clear the calculation
	c = cell(sys.argv[1])
	clear_calc()

if False:
	# Perturb the first atom a little (for testing)
	disp = []
	for i in range(0, len(c.atoms)):
		disp.append(np.zeros(3))
	disp[0] = [0.1,0,0]
	c.perturb_atoms_cart(disp)

for i in range(0,100):

	c.run_singlepoint_qe(label="relax")
	
	# Due to translational invariance, we may
	# keep the first atom fixed
	pert = c.forces_cart
	pert[0] = [0,0,0]

	# Simply follow the force downhill
	c.perturb_atoms_cart(pert)
	c.apply_strain_cart(c.stress_cart/1000)

	# Stop if converged
	if c.total_force  > 10e-8: continue
	if c.total_stress > 0.01: continue
	break
