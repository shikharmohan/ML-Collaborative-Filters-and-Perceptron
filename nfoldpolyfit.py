#	Starter code for linear regression problem
#	Below are all the modules that you'll need to have working to complete this problem
#	Some helpful functions: np.polyfit, scipy.polyval, zip, np.random.shuffle, np.argmin, np.sum, plt.boxplot, plt.subplot, plt.figure, plt.title
import sys
import csv
import math
import random
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pdb
import copy

def nfoldpolyfit(X, Y, maxK, n, verbose):
#	NFOLDPOLYFIT Fit polynomial of the best degree to data.
#   NFOLDPOLYFIT(X,Y,maxDegree, nFold, verbose) finds and returns the coefficients 
#   of a polynomial P(X) of a degree between 1 and N that fits the data Y 
#   best in a least-squares sense, averaged over nFold trials of cross validation.
#
#   P is a vector (in numpy) of length N+1 containing the polynomial coefficients in
#   descending powers, P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1). use
#   numpy.polyval(P,Z) for some vector of input Z to see the output.
#
#   X and Y are vectors of datapoints specifying  input (X) and output (Y)
#   of the function to be learned. Class support for inputs X,Y: 
#   float, double, single
#
#   maxDegree is the highest degree polynomial to be tried. For example, if
#   maxDegree = 3, then polynomials of degree 0, 1, 2, 3 would be tried.
#
#   nFold sets the number of folds in nfold cross validation when finding
#   the best polynomial. Data is split into n parts and the polynomial is run n
#   times for each degree: testing on 1/n data points and training on the
#   rest.
#
#   verbose, if set to 1 shows mean squared error as a function of the 
#   degrees of the polynomial on one plot, and displays the fit of the best
#   polynomial to the data in a second plot.
#   
#
#   AUTHOR: Shikhar Mohan
	chunkX = int(len(X)/n)
	fitval = []
	plot1 = []
	partX = []
	partY = []
	end = chunkX
	start = 0
	for i in range(1,n+1):
		partX.append(X[start:end])
		partY.append(Y[start:end])
		start = chunkX*i
		end = chunkX*(i+1)

	for j in range(0, maxK+1):
		print "k: ", j
		runs = []
		for i in range(0, n):
			testX = partX.pop(i)
			testY = partY.pop(i)
			vectorX = convert1D(partX)
			vectorY = convert1D(partY)
			P = np.polyfit(vectorX, vectorY, j)
			trained = np.polyval(P, testX)
			mse = meanSquare(trained, testY)
			runs.append(mse)
			print "Kth Order : ", j
			print "cross valid# ", i
			print runs
			partX.insert(i, testX)
			partY.insert(i, testY)

		avg_mse = 0
		for r in runs:
			avg_mse += r
		avg_mse = float(avg_mse)/len(runs)
		print avg_mse
		plot1.append((j, avg_mse))

	bestModel = [-1, float("inf")]
	mses =[]
	for p in plot1:
		if p[1] < bestModel[1]:
			bestModel = p
	mses = []
	for m in plot1:
		mses.append(m[1])

	bestpoly = np.polyfit(X, Y, bestModel[0])

	if verbose == 1:
		plt.figure(1)
		plt.subplot(211)
		plt.plot(mses)
		plt.ylabel("Mean Squared Errors")
		plt.xlabel("Degrees")
		plt.subplot(212)
		t = np.arange(-1.0, 1, 0.01)
		test = np.polyval(bestpoly,t)
		plt.plot(X,Y, 'r^')
		plt.plot(t, test, 'g')
		print "Best Model is K: ", bestModel[0]
		print bestpoly
		print plot1
		plt.show()

	return bestModel[0]





def convert1D(arr):
	a = []
	for i in range(0,len(arr)):
		for j in range(0, len(arr[0])):
			a.append(arr[i][j])
	return a

def meanSquare(inp, out):
	total = []
	for i in range(0, min(len(inp), len(out))):
		diff = inp[i] - out[i]
		square = diff*diff;
		total.append(square)
	#calculate average of squares
	s = 0;
	for i in range(0, len(total)):
		s += total[i]
	result = float(s)/len(total)
	return result


def main():
	# read in system arguments, first the csv file, max degree fit, number of folds, verbose
	rfile = sys.argv[1]
	maxK = int(sys.argv[2])
	nFolds = int(sys.argv[3])
	verbose = bool(sys.argv[4])
	
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []
	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):
		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)
	nfoldpolyfit(X, Y, maxK, nFolds, verbose)

if __name__ == "__main__":
	main()
