import numpy as np
import os

vol_per_atom = 9.8273305

def cell(y,z):
	global vol_per_atom
	vol = y*z
	sf = (2.0*vol_per_atom/vol)**(1.0/3.0)
	x = sf
	y = y*sf
	z = z*sf
	print "a,b,c: ",x,y,z
	print "Cell volume:", x*y*z, "A^3"
	ret  = "%block lattice_cart\n"
	ret +=  str(x) + " 0 0\n"
	ret += "0 "    + str(y) + " 0\n"
	ret += "0 0 "  + str(z) + "\n"
	ret += "%endblock lattice_cart\n\n"
	ret += "%block positions_frac\n"
	ret += "Fe 0   0   0\n"
	ret += "Fe 0.5 0.5 0.5\n"
	ret += "%endblock positions_frac\n\n"
	ret += "kpoint_mp_spacing 0.05\n"
	ret += "symmetry_generate"
	return ret

def param(y,z):
	ret  = "task singlepoint\n"
	ret += "cut_off_energy 500 eV\n"
	ret += "nextra_bands 10\n"
	ret += "write_none true\n"
	return ret

GRID_SIZE = 10

os.system("rm *.castep")

for y in np.linspace(0.5,2,GRID_SIZE+1):
	for z in np.linspace(0.5,2,GRID_SIZE+1):
		seed = str(y)+"_"+str(z)
		cellf  = open(seed+".cell", "w")
		paramf = open(seed+".param","w")
		cellf.write(cell(y,z))
		paramf.write(param(y,z))
		cellf.close()
		paramf.close()
		os.system("nice -15 mpirun castep.mpi "+seed)
		os.system("rm "+seed+".cell")
		os.system("rm "+seed+".param")
