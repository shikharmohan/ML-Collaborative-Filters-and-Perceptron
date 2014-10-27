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
	distance = int(sys.argv[2])
	kNN = int(sys.argv[3])
	ignore = int(sys.argv[5])
	if kNN <= 0: 
		kNN = 1
	item = int(sys.argv[4])
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter='\t')
	data = []
	for i, row in enumerate(dat):
		if i > 0:
			data.append((int(row[0]), int(row[1]), int(row[2])))
	matrix = createMatrix(item, dat)
	chunk = 2000
	end = chunk
	start = 0
	partition = []
	ttest = []
	for i in range(1,51):
		partition.append(data[start:end])
		start = chunk*i
		end = chunk*(i+1)
	#100-fold cross valid
	for i in range(0, 50):
		test = partition.pop(i)
		random.shuffle(test)
		test_matrix = createPartMatrix(test[:100], item)
		train_flat = [js for sublist in partition for js in sublist]
		train = createPartMatrix(train_flat, item)
		error = getScore(test_matrix, test[:100], train, distance, item, ignore, kNN)
		ttest.append(error)
		partition.insert(i, test)

	print ttest

def getScore(test_matrix, test, train, distance, item, ignore, k):
	error = 0
	if item == 1:
		#item_cf
		for t in test:
			curr = test_matrix[t[1]]
			topDist = []
			if distance == 0:
				#pearson
				for i in train.keys():
					if i != t[1] and train[i][t[0]] != 0 and ignore == 0:
						array1 = list(curr.values())
						array2 = list(train[i].values())
						topDist.append((i, 1 - scipy.stats.pearsonr(array1, array2)[0]))
					elif i != t[1] and ignore == 1:
						#include zeros
						array1 = list(curr.values())
						array2 = list(train[i].values())
						currDist = scipy.spatial.distance.cityblock(array1, array2)
						topDist.append((i, currDist))
			else:
				#manhattan
				for i in train.keys():
					if i != t[1] and train[i][t[0]] != 0 and ignore == 0:
						array1 = list(curr.values())
						array2 = list(train[i].values())
						currDist = scipy.spatial.distance.cityblock(array1, array2)
						topDist.append((i, currDist))
					elif i != t[1] and ignore == 1:
						#include zeros
						array1 = list(curr.values())
						array2 = list(train[i].values())
						currDist = scipy.spatial.distance.cityblock(array1, array2)
						topDist.append((i, currDist))

			topDist.sort(key=operator.itemgetter(1))
			selectedK = []
			if k > len(topDist):
				k = len(topDist)
			for js in range(0, k):
				selectedK.append(train[topDist[js][0]][t[0]])
			if len(selectedK) == 0:
				prediction = 3
			else:
				prediction = max(set(selectedK), key=selectedK.count)
			if prediction != t[2]:
				print prediction, ",", t[2]
				error += 1
			else:
				print "Got it right!"
	else:
		#user_cf
		for t in test:
			curr = test_matrix[t[0]]
			topDist = []
			if distance == 0:
				#pearson
				for i in train.keys():
					if i != t[0] and train[i][t[1]] != 0 and ignore == 0:
						array1 = list(curr.values())
						array2 = list(train[i].values())
						currDist = 1 - scipy.stats.pearsonr(array1, array2)[0]
						topDist.append((i, currDist))
					elif i != t[0] and ignore == 1:
						array1 = list(curr.values())
						array2 = list(train[i].values())
						currDist = scipy.spatial.distance.cityblock(array1, array2)
						topDist.append((i, currDist))
			else:
				#manhattan
				for i in train.keys():
					if i != t[0] and train[i][t[1]] != 0 and ignore == 0:
						array1 = list(curr.values())
						array2 = list(train[i].values())
						currDist = scipy.spatial.distance.cityblock(array1, array2)
						topDist.append((i, currDist))
					elif i != t[0] and ignore == 1:
						array1 = list(curr.values())
						array2 = list(train[i].values())
						currDist = scipy.spatial.distance.cityblock(array1, array2)
						topDist.append((i, currDist))					

			topDist.sort(key=operator.itemgetter(1))
			selectedK = []
			if k > len(topDist):
				k = len(topDist)
			for js in range(0, k):
				selectedK.append(train[topDist[js][0]][t[1]])
			if len(selectedK) == 0:
				prediction = 3
			else:
				prediction = max(set(selectedK), key=selectedK.count)
			if prediction != t[2]:
				print prediction, ",", t[2]
				error += 1
			else:
				print "Got it right!"
	print "Error : ", error
	return error

def createPartMatrix(x, item):
	if item == 1:
		#Movies 
		u = tuple(i[0] for i in x)
		m = tuple(i[1] for i in x)
		r = tuple(i[2] for i in x)
		users = range(1,944)
		moviesID = set(m)

		movieDict = dict(zip(users, [0]*len(users)))
		matrix = {}
		for i in moviesID:
			matrix[i] = copy.deepcopy(movieDict)

		for j in range(0, len(x)):
			userkey = copy.deepcopy(x[j][0])
			moviekey = copy.deepcopy(x[j][1])
			rating = copy.deepcopy(x[j][2])

			matrix[moviekey][userkey] = copy.deepcopy(rating)
	else:
		u = tuple(i[0] for i in x)
		m = tuple(i[1] for i in x)
		r = tuple(i[2] for i in x)
		users = set(u)
		moviesID = range(1,1683)

		movieDict = dict(zip(moviesID, [0]*len(moviesID)))
		matrix = {}
		for i in users:
			matrix[i] = copy.deepcopy(movieDict)

		for j in range(0, len(x)):
			userkey = copy.deepcopy(x[j][0])
			moviekey = copy.deepcopy(x[j][1])
			rating = copy.deepcopy(x[j][2])

			matrix[userkey][moviekey] = copy.deepcopy(rating)

	return matrix	


def createMatrix(item, dat):
	if item == 1:
		#movie
		data = []
		u = []
		m = []
		for i, row in enumerate(dat):
			if i > 0:
				m.append(int(row[1]))
				data.append((int(row[0]), int(row[1]), int(row[2])))

		#keys: users, 
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
	else:
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

	return matrix


main()

