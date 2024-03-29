import sys
import csv
import numpy as np
import scipy


# w is the parameter vector (weight vector and threshold value)
# m is the number of training examples
# xk is the kth training example
# yk is the kth training example class label {-1,1}

def perceptrona(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.
	k = 0
	#weights w0 and w1
	weights = w_init
	data = [((1,xk), yk) for xk, yk in zip(X,Y)]
	while True:
		#tracks number of missclassified training examples
		missclassified = 0
		
		for x, yk in data:
			val = np.dot(x, weights)
			if val > 0:
				cl = 1
			elif val < 0:
				cl = -1
			else:
				cl = 0
			if yk != cl:
				missclassified += 1
				weights = [weight + xk * yk for weight, xk in zip(weights, x)]
		if missclassified == 0:
			break
		k += 1
	print (weights,k)	
	return (weights, k)

def main():
	rfile = sys.argv[1]
	
	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
	w_init = np.array([0,0])
	perceptrona(w_init, X1, Y1)
	perceptrona(w_init, X2, Y2)

if __name__ == "__main__":
	main()
