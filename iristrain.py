import neural
import numpy as np
import random

data = []

f = open('irisdatatrain.txt')
for line in f.readlines():
	l = line.split(',')
	data.append(list(map(float, l)))

wts, b = neural.nn(data, 0.5)

test = []

f = open('irisdatatrain.txt')
for line in f.readlines():
	l = line.split(',')
	test.append(list(map(float, l)))

for x in range(int(len(test) / 2)):
	i = int(random.uniform(1, len(test)))
	prediction = neural.sigmoid(np.sum(wts * test[i][:-1]) + b)
	print(int(test[i][-1]), int(round(prediction)))
