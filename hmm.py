import numpy as np
import logging
import sys

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
				tag = content[1].split('\n')[0]
				trans[previous+"_"+tag] += 1
				context[tag] += 1
				emit[tag+"_"+token] += 1
				previous = tag
			else:
				trans[previous+'_</s>'] += 1
				previous = '<s>'
				context[previous] += 1

		self.logger.info("Printing transitions")
		# for key in trans:
		# 	prob = trans[key]/context[key.split('_')[0]]
		# 	if prob > 1:
		# 		print key, prob
		# 		sys.exit(1)
		# 	self.logger.info("Transition:" + str(key) + ": " + str(trans[key]/context[key.split('_')[0]]))

		# for key in emit:
		# 	if prob > 1:
		# 		print key, prob
		# 		sys.exit(1)
		# 	self.logger.info("Emission:" + str(key) + ": " + str(emit[key]/context[key.split('_')[0]]))

		self.theta = [context,trans,emit]

	def test(self,test_file):
		self.logger.info("Started HMM Testing")

		previous = '<s>'

		context = self.theta[0]
		trans = self.theta[1]
		emit = self.theta[2]

		tags = context.keys()

		prev_prob = 1.0

		f = open('data/pred.col','w')


		for line in open(test_file):
			content = line.split('\t',1)
			if line != '\n':
				token = content[0]

				curr = [1] * len(tags)


				for i,tag in enumerate(tags):
					curr[i] *= prev_prob * (trans[previous+"_"+tag]/context[previous]) * (emit[tag+"_"+token]/context[tag])


				curr = np.asarray(curr)
				prev_prob = np.max(curr)
				previous = tags[np.argmax(curr)]


				f.write(token + "\t" + tags[np.argmax(curr)] + "\n")

			else:
				prev_prob = 1.0
				f.write('\n')
				previous = '<s>'

		f.close()
		self.logger.info("Done Testing, results are written in pred.col")


	def get_theta(self):
		return self.theta


	def load_theta(self, theta):
		self.theta = theta