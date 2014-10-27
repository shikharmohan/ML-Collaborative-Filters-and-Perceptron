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
		m.append(int(row[1]))
		data.append(( int(row[0]), int(row[1]), int(row[2])))

users = range(1,944)
moviesID = set(m)

movieDict = dict(zip(users, [0]*len(users)))
matrix = {}
for i in moviesID:
	matrix[i] = copy.deepcopy(movieDict)

for j in range(0, len(data)):
	userkey = copy.deepcopy(data[j][0])
	moviekey = copy.deepcopy(data[j][1])
	rating = copy.deepcopy(data[j][2])

	matrix[moviekey][userkey] = copy.deepcopy(rating)

result = []

for i in matrix.keys():
	rev=0
	for j in matrix[i].keys():
		if matrix[i][j] != 0:
			rev += 1
	result.append(rev)

result.sort(reverse=True)

print "Most: ", result[0]
print "Least: ", result[len(result)-1]
x = range(1,1683)
plt.figure()
plt.title("3B")
plt.xlabel("Movie's Number")
plt.ylabel("Number of Reviews")
plt.plot(x,result, 'r^')
plt.show()

