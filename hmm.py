import numpy as np
import logging

from collections import defaultdict

class HMM(object):

	def __init__(self):
		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)
		self.logger.info("Initializing HMM")


	def train(self, train_file):

		self.logger.info("Started HMM Training")


		context = defaultdict(float)
		trans = defaultdict(float)
		emit = defaultdict(float)
		previous = '<s>'
		context[previous]+= 1

		for line in open(train_file):

			content = line.split('\t',1)

			if line != '\n':
				token = content[0]
				# print content
				tag = content[1].split('\n')[0]
				trans[previous+"_"+tag] += 1
				context[tag] += 1
				emit[tag+"_"+token] += 1
				previous = tag
			else:
				trans[previous+'_</s>'] += 1
				previous = '<s>'
		self.logger.info("Printing transitions")
		for key in trans:
			self.logger.info("Transition:" + str(key) + ": " + str(trans[key]/context[key.split('_')[0]]))

		self.theta = [context,trans,emit]

	def get_theta(self):
		return self.theta


	def load_theta(self, theta):
		self.theta = theta