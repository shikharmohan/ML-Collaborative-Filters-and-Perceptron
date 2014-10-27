import sys
import csv
import math
import random
import numpy as np
import scipy
import scipy
import matplotlib.pyplot as plt
import pdb
import copy
import operator
from scipy.spatial.distance import cityblock
from scipy.stats import mode

rfile = sys.argv[1]
csvfile = open(rfile, 'rb')
dat = csv.reader(csvfile, delimiter='\t')

data = []
u = []
m = []
for i, row in enumerate(dat):
	if i > 0:
		u.append(int(row[0]))
		data.append(( int(row[0]), int(row[1]), int(row[2])))

users = set(u)
moviesID = range(1,1683)

movieDict = dict(zip(moviesID, [0]*len(moviesID)))
matrix = {}
for i in users:
	matrix[i] = copy.deepcopy(movieDict)

for j in range(0, len(data)):
	userkey = copy.deepcopy(data[j][0])
	moviekey = copy.deepcopy(data[j][1])
	rating = copy.deepcopy(data[j][2])

	matrix[userkey][moviekey] = copy.deepcopy(rating)

counts = []
heights = []

for i in range(1,943):
	for j in range(i+1, 943):
		ct = 0
		for k in range(1,1683):
			if matrix[j][k] != 0 and matrix[i][k] != 0:
				ct += 1
		heights.append(ct)
		counts.append((i,j,ct))

print "Len: ,", len(heights)

print "Heights ", heights
print "Median: ", np.median(np.array(heights))
print "Mean: ", np.mean(np.array(heights))
plt.figure()
plt.ylabel("Number of User Pairs")
plt.xlabel("Number of Movies Reviewed")
plt.hist(heights, 20)
plt.show()



