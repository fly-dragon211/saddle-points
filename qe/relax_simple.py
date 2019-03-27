from cell import cell
import sys

c = cell(sys.argv[1])

for i in range(0,100):

	c.run_singlepoint_qe(sys.argv[1])
	
	# Due to translational invariance, we may
	# keep the first atom fixed
	pert = c.forces_cart
	pert[0] = [0,0,0]

	# Simply follow the force downhill
	c.perturb_atoms_cart(pert)
	c.apply_strain_cart(c.stress_cart/1000)
