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

def main():
	rfile = sys.argv[1]
	userID = int(sys.argv[2])
	movieID = int(sys.argv[3])
	distance = int(sys.argv[4])
	kNN = int(sys.argv[5])
	if kNN <= 0: 
		kNN = 1
	item = int(sys.argv[6])
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter='\t')

	data = []
	u = []
	m = []
	for i, row in enumerate(dat):
		if i > 0:
			u.append(int(row[0]))
			data.append(( int(row[0]), int(row[1]), int(row[2])))

	#keys: users, 
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
	if item == -1:
		cacheDistances(matrix)
	else:
		predictedRating = getScore(matrix, userID, movieID, distance, kNN, i)
		print "UserID: ", userID, " movieID: ", movieID, " trueRating: ", matrix[userID][movieID], " predictedRating: ", predictedRating, " distance: ", distance, " K: ",kNN, " i: ", item 

def cacheDistances(matrix):
	output = [[0 for x in xrange(943)] for x in xrange(943)]
	for u in range(1, 944):
		curr = matrix[u]
		for i in range(1, len(matrix)):
			if i != u:
				array1 = list(curr.values())
				array2 = list(matrix[i].values())
				pearson = 1 - scipy.stats.pearsonr(array1, array2)[0]
				dist = scipy.spatial.distance.cityblock(array1, array2)
				output[u-1][i-1] = (pearson, dist)
	print output

def getScore(matrix, user, movie, dist, k, ignore):
	curr = copy.deepcopy(matrix[user])
	topDist = []
	if dist == 0:
		#pearson
		for i in range(1, len(matrix)):
			if i != user:
				array1 = list(curr.values())
				array2 = list(matrix[i].values())
				currDist = 1 - scipy.stats.pearsonr(array1, array2)[0]
				topDist.append((i, currDist))
	else:
		#manhattan
		for i in range(1, len(matrix)):
			if i != user:
				array1 = list(curr.values())
				array2 = list(matrix[i].values())
				currDist = scipy.spatial.distance.cityblock(array1, array2)
				topDist.append((i, currDist))

	topDist.sort(key=operator.itemgetter(1))
	print "Top Dist: ", topDist
	selectedK = []
	if ignore == 1: 
		selectedK = copy.deepcopy(topDist[:k]) #pick top K
	else:
		#no zeroes
		count = 0
		it = 0
		while count < k:
			if matrix[topDist[it][0]][movie] == 0:
				it += 1
				continue
			else:
				selectedK.append(matrix[topDist[it][0]][movie])
				count += 1
				it += 1
		print "SelectedK: ", selectedK
	return max(set(selectedK), key=selectedK.count)


main()

