from collections import defaultdict
import random

class Perceptron(object):
	def __init__(self):
		self.weights = defaultdict(float)
		self.bias = 0.1

	def inc_weight(self,keys,step):
		for i in keys:
			self.weights[i] += 1 * step
		self.bias += 0.1

	def dec_weight(self,keys,step):
		for i in keys:
			self.weights[i] -= 1 * step
		self.bias -= 0.1

	def activate(self,keys):
		result = 0
		for i in keys:
			result += self.weights[i] + ((self.weights[i] == 0) * random.uniform(0, 1))
		return self.sign(result + self.bias)


	def sign(self,val):
		return (val > 0) * 2 - 1