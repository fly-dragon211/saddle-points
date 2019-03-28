import os
import sys

if len(sys.argv) > 1:
	num_req = int(sys.argv[1])
else:
	num_req = -1

max_n = -1
to_save = None
for f in os.listdir("singlepoints"):
	if not f.endswith(".out"): continue
	num = int(f.split("_")[-1].split(".")[0])
	if num == num_req:
		to_save = f
		break
	if num > max_n:
		max_n = num
		to_save = f

max_n = -1
for f in os.listdir("saved"):
	if not f.endswith(".out"): continue
	num = int(f.split("_")[-1].split(".")[0])
	if num > max_n:
		max_n = num

os.system("cp singlepoints/"+to_save+" saved/"+str(max_n+1)+".out")
