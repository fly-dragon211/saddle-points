import os
import sys

if len(sys.argv) > 1:
	num_req = int(sys.argv[1])
else:
	num_req = -1

max_n = -1
flast = None
for f in os.listdir("singlepoints"):
	if not f.endswith(".out"): continue
	if not "JOB DONE" in open("singlepoints/"+f).read(): continue
	num = int(f.split("_")[-1].split(".")[0])
	if num == num_req:
		flast = f
		break
	if num > max_n:
		max_n = num
		flast = f

scr = open("jmol_script","w")
scr.write("load singlepoints/"+flast+" {2 2 2}")
scr.close()

os.system("jmol -s jmol_script")
os.system("rm jmol_script")
