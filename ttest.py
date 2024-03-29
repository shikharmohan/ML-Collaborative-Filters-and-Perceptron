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

inp1810 =[63, 74, 74, 66, 76, 74, 75, 72, 63, 69, 72, 74, 77, 63, 61, 68, 57, 72, 71, 71, 71, 67, 69, 70, 71, 66, 67, 69, 70, 77, 65, 68, 79, 67, 79, 70, 68, 71, 76, 63, 73, 67, 62, 77, 66, 71, 70, 69, 76, 71]
inp0810 = [66, 66, 65, 62, 65, 66, 64, 64, 73, 65, 59, 64, 53, 61, 77, 62, 74, 72, 67, 60, 69, 65, 62, 59, 65, 62, 63, 65, 64, 66, 61, 66, 63, 70, 59, 65, 56, 70, 60, 65, 62, 78, 69, 57, 58, 74, 69, 65, 69, 63]

#6D
inp1801 = [100, 100, 99, 100, 98, 100, 99, 100, 100, 95, 100, 100, 100, 100, 97, 100, 100, 100, 98, 99, 100, 100, 99, 100, 100, 98, 100, 100, 99, 100, 100, 99, 100, 99, 100, 96, 100, 100, 100, 99, 100, 100, 99, 100, 99, 100, 100, 99, 100, 98]
inp1800 = [65, 72, 72, 71, 68, 76, 79, 75, 72, 75, 65, 65, 76, 72, 76, 67, 66, 70, 76, 72, 72, 67, 67, 59, 80, 66, 71, 76, 81, 70, 69, 71, 76, 67, 74, 61, 66, 68, 66, 70, 62, 71, 72, 81, 80, 80, 73, 69, 64, 72]
inp0800 = [70, 70, 67, 69, 72, 61, 62, 72, 73, 74, 60, 72, 71, 74, 62, 70, 86, 63, 74, 69, 69, 80, 72, 74, 72, 75, 74, 67, 70, 57, 72, 72, 68, 68, 62, 69, 69, 72, 69, 72, 72, 71, 63, 65, 63, 72, 69, 70, 73, 68]
inp0801 = [100, 100, 98, 100, 98, 100, 99, 100, 100, 95, 100, 100, 99, 100, 100, 95, 99, 100, 98, 99, 100, 100, 99, 100, 100, 98, 100, 96, 99, 100, 100, 96, 100, 99, 100, 96, 100, 100, 100, 99, 100, 100, 100, 100, 99, 99, 100, 99, 100, 98]

k1 = [58, 71, 72, 71, 65, 75, 73, 66, 80, 75, 69, 64, 60, 69, 69, 66, 70, 73, 66, 74, 70, 81, 63, 72, 68, 71, 72, 70, 76, 73, 81, 77, 75, 76, 67, 76, 73, 67, 78, 83, 76, 68, 67, 76, 65, 66, 67, 68, 76, 73]
k2 = [77, 60, 71, 62, 69, 63, 68, 64, 76, 78, 63, 63, 66, 62, 73, 73, 70, 71, 70, 76, 73, 68, 62, 70, 67, 69, 75, 60, 76, 67, 66, 69, 68, 67, 63, 69, 68, 70, 69, 61, 71, 63, 71, 74, 73, 69, 74, 73, 68, 62]
k4 = [68, 72, 78, 74, 70, 75, 60, 76, 80, 68, 75, 75, 62, 63, 70, 73, 70, 73, 71, 63, 60, 62, 69, 69, 79, 66, 76, 61, 75, 73, 71, 69, 71, 72, 70, 72, 78, 73, 64, 67, 69, 73, 64, 70, 66, 76, 64, 74, 69, 65]
k8 = [70, 70, 67, 69, 72, 61, 62, 72, 73, 74, 60, 72, 71, 74, 62, 70, 86, 63, 74, 69, 69, 80, 72, 74, 72, 75, 74, 67, 70, 57, 72, 72, 68, 68, 62, 69, 69, 72, 69, 72, 72, 71, 63, 65, 63, 72, 69, 70, 73, 68]
k16 = [68, 75, 71, 65, 70, 68, 73, 64, 66, 71, 64, 62, 71, 66, 78, 77, 72, 69, 63, 72, 70, 69, 65, 70, 79, 74, 67, 72, 70, 70, 71, 72, 70, 73, 65, 70, 74, 69, 66, 68, 71, 74, 64, 72, 68, 73, 68, 66, 76, 72]
k32 = [68, 74, 70, 65, 62, 69, 72, 58, 70, 57, 71, 77, 71, 71, 65, 73, 63, 74, 61, 71, 63, 74, 66, 73, 73, 77, 67, 69, 66, 65, 73, 72, 74, 62, 63, 69, 62, 70, 72, 67, 64, 73, 66, 75, 70, 68, 67, 65, 79, 74]

ki32 = [67, 55, 67, 65, 66, 66, 66, 64, 60, 67, 66, 65, 68, 66, 59, 57, 63, 66, 56, 61, 60, 61, 61, 61, 54, 60, 61, 60, 59, 61, 64, 67, 67, 59, 66, 58, 64, 66, 59, 63, 63, 58, 69, 69, 62, 70, 68, 66, 67, 65]


print scipy.stats.ttest_ind(ki32, k32)

avk1 = float(sum(k1)/len(k1))
avk2 = float(sum(k1)/len(k1))
avk4 = float(sum(k1)/len(k1))
avk8 = float(sum(k1)/len(k1))
avk16 = float(sum(k1)/len(k1))
avk32 = float(sum(k1)/len(k1))

plotk = [(1,avk1), (2, avk2), (4, avk4), (8, avk8), (16, avk16), (32, avk32)]


plt.figure()
plt.title("Item K32")
plt.ylabel("Frequency of Error Counts")
plt.xlabel("Error Count out of 100")
plt.xlim(50,100)
plt.hist(ki32, 20)
# plt.subplot(212)
# plt.title("Manhattan & Non-Zero")
# plt.ylabel("Frequency of Error Counts")
# plt.xlabel("Error Count out of 100")
# plt.xlim(50,100)
# plt.hist(inp1800, 20)
# plt.subplot(211)
# plt.title("Pearson & Zero")
# plt.ylabel("Frequency of Error Counts")
# plt.xlabel("Error Count out of 100")
# plt.hist(inp0801, 20)
# plt.xlim(50,100)
# plt.subplot(212)
# plt.title("Pearson & Non-Zero")
# plt.ylabel("Frequency of Error Counts")
# plt.xlabel("Error Count out of 100")
# plt.xlim(50,100)
# plt.hist(inp0800, 20)


# plt.title("K vs Avg Error")
# plt.ylabel("Frequency of Error Counts")
# plt.xlabel("Error Count out of 100")
# plt.plot(plotk)


plt.show()

