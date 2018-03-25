import numpy as np
from matplotlib import pyplot as plt

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def dsigmoid(x):
	return sigmoid(x) * (1 - sigmoid(x))

def nn(data, learnrate):	
	costs = []
	wts = np.array([np.random.randn() for x in range(len(data[0]) - 1)])
	b = np.random.randn()
	for i in range(50000):
		randind = np.random.randint(len(data))
		point = data[randind]
		z = b + np.sum(wts * point[:-1])
		prediction = sigmoid(z)
	
		target = point[-1]
		cost = np.square(prediction - target)
		
		dcost_dpred = 2 * (prediction - target)
		dpred_dz = dsigmoid(z)
		dz_dw = np.array(point[:-1])
		dz_db = 1
		dcost_dz = dcost_dpred * dpred_dz
		dcost_dw = np.array(dcost_dpred * dpred_dz * dz_dw)
		dcost_db = dcost_dz * dz_db
		
		wts = wts - learnrate * dcost_dw
		b = b - learnrate * dcost_db
	
	return wts, b






