import numpy as np
import logging
import sys


class LogisticRegression(object):

	def __init__(self, meta_data):

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)

		self.logger.info("Initializing Logistic Regression")

		self.alpha = 0.0005
		self.reg = 10

		self.theta = []
		self.meta_data = meta_data

		self.vocab = meta_data[0]
		self.labels = meta_data[1]
		self.suffixes = meta_data[2]
		self.feat_size = 4

		self.dict_labels = {}
	#  sigmoid function

	def sigmoid(self,X):
		return 1 / (1 + np.exp(-X))

	# cost function

	def cost(self,theta, X,y):
		hyp = self.sigmoid(np.dot(X, theta))

		cost = -y * np.log(hyp) - (1 - y) * np.log(1-hyp)
		return cost.mean() + (self.reg/(2 * X.shape[0]) * sum(np.power(theta,2)))


	# gradient descent

	def grad(self, X, inst_vals,inst_labels):

		for count, label in enumerate(self.labels):

			curr_theta = self.theta[count]
			
			indices = np.array(self.dict_labels[label])

			y = np.ones(len(inst_vals)) * -1

			y[indices] *= -1

			for i in range(500):

				hyp = self.sigmoid(np.dot(X, curr_theta))

				error = hyp - y
				current_cost = self.cost(curr_theta, X,y)

				if current_cost > 0:

					theta_0 = curr_theta[0]
					curr_theta = curr_theta - (self.alpha * (np.dot(error,X) + ((self.reg / X.shape[0]) * curr_theta)))
					if self.reg > 0:
						curr_theta[0] = theta_0

					self.logger.debug("Label: " + str(count) + " out of: " + str(len(self.labels)) + ", Iteration: " + str(i) + ", Cost: " + str(current_cost))

			self.theta[count] = curr_theta


	def train(self,feat_tuple):
		self.logger.info("Started training Logistic Regression")

		feat_idx = feat_tuple[0]
		inst_labels = feat_tuple[1]
		inst_vals = feat_tuple[2]
		self.dict_labels = feat_tuple[3]


		self.theta = 0.01 * np.random.randn(len(self.labels),self.feat_size)
		self.grad(feat_idx, inst_vals,inst_labels)


	def test(self,feat_tuple):
		self.logger.info("Started testing Perceptron")

		feat_idx = feat_tuple[0]
		inst_labels = feat_tuple[1]
		inst_vals = feat_tuple[2]

		f = open('data/pred.col','w')

		for count, instance in enumerate(feat_idx):
			self.logger.debug("Testing instance: " + str(count))


			if not inst_vals[count]:                
				f.write('\n')
				continue
			else:
				
				hyp = self.sigmoid(np.dot(self.theta,instance))
				print hyp
				f.write(inst_vals[count] + "\t" + self.labels[np.argmax(hyp)] + "\n")

		f.close()
		self.logger.info("Done Testing, results are written in pred.col")

	def get_theta(self):
		return self.theta


	def load_theta(self, theta):
		self.theta = theta