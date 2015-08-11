from collections import defaultdict
import random
import numpy as np
import sys
class Perceptron(object):
	def __init__(self):
		self.weights = defaultdict(float)
		self.bias = 0.1

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
			result += self.weights[i] + ((self.weights[i] == 0) * random.uniform(0, 1))
		if test:
			return result + self.bias
		else:
			return self.sign(result + self.bias)


	def sign(self,val):
		return (val > 0) * 2 - 1