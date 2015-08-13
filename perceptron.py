from collections import defaultdict
import random
import numpy as np
import sys
class Perceptron(object):
	def __init__(self,keys):

		values = np.random.uniform(0,1,(len(keys)))

		self.weights = dict(zip(keys,values))
		self.bias = 0.1
		self.history = []
	def inc_weight(self,feats,step):
		for i in feats:
			self.weights[i] += 1 * step
		self.bias += 0.1

	def dec_weight(self,feats,step):
		for i in feats:
			self.weights[i] -= 1 * step
		self.bias -= 0.1


	def activate(self,feats,test):
		result = 0
		for i in feats:
			result += self.weights[i]
		return result + self.bias


	def save_iter(self):
		self.history.append(self.weights.values())

	def average(self):
		arr = np.asarray(self.history)
		avg = np.average(arr, axis=0)
		keys = self.weights.keys()

		self.weights = dict(zip(keys,avg))