from cell import cell, last_cell, clear_calc
import numpy as np
import sys
import os

# Get a set of random pertubations to the atoms
# that form a normalized vector in configuration space
def get_rand_pert(num_atoms):
	disp  = 2*(np.random.rand(num_atoms*3) - 0.5)
	disp /= np.linalg.norm(disp)
	ret   = []
	for i in range(0, num_atoms):
		pert = []
		for j in range(0,3):
			pert.append(disp[i*3+j])
		ret.append(pert)
	return np.array(ret)

# Returns a pertubation in the same direction
# as pert, with the maximum atom displacement given
def set_max_disp(pert, max_disp):
	return pert * max_disp / max(np.linalg.norm(p) for p in pert)

# Returns the dot product of two pertubations
# in configuration space
def dot_pert(p1, p2):
	return np.dot([x for r in p1 for x in r],
		      [x for r in p2 for x in r])


# Load the cell and run an initial singlepoint (for comparision)
if len(sys.argv) < 2: 
	# Load the cell from the last singlepoint
	c = last_cell()
else:
	# Load the specified cell and clear the calculation
	clear_calc()
	c = cell(sys.argv[1])
c.run_singlepoint_qe()

# Pick the direction in configuration
# space and perturb the cell along it
norm = get_rand_pert(len(c.atoms))
c.perturb_atoms_cart(set_max_disp(norm, 0.1))

# Run activation-relaxation
while True:

	# Calculate forces etc
        c.run_singlepoint_qe(label="saddle_point_search")

	# Due to translational invariance
	# we may keep the first atom position fixed
        pert = c.forces_cart
        pert[0] = [0,0,0]

	# Work out the parallel and perpendicular
	# components of our pertubation to norm
	pert_para = dot_pert(pert, norm) * norm
	pert_perp = pert - pert_para

	# Invert the paralle one and apply the pertubation
        c.perturb_atoms_cart(pert_perp - pert_para)

	# Relax the lattice vectors
        c.apply_strain_cart(c.stress_cart/1000)

	# Stop if converged
	if c.total_force  > 10e-8: continue
        if c.total_stress > 0.01: continue
        break

