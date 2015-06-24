import numpy as np
from scipy.sparse import csr_matrix

import re
import logging
import sys
import time

'''
Features:
vocab id
common suffix id
is digit
is capital

'''
class FeatureSet(object):


	def __init__(self, meta_data):

		self.reg = re.compile("^\d+((\,|\.|\/)\d+)*$")
		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)

		self.vocab = meta_data[0]
		self.labels = meta_data[1]




	# FEATURES:
	# word form (N)
	# common prefixes (P)
	# common suffixes (S)
	# is upper (1)
	# is capital (1)
	# is number (1)
	############################
	# token length (len(max(N)))
	# token position (len(max(Sentences)))
	# form of previous and next tokens (N*2)
	# length of previous and next tokens (len(max(N))*2)


	def extract_feats(self, file_path):


		self.logger.info("Started Feature Extraction")

		self.logger.info("Extracting tokens and sentences")
		# get all tokens in the training file
		with open(file_path) as file:
			tokens = np.array([line.strip().decode('utf-8').split('\t',1)[0] for line in file])

		with open(file_path) as file:
			labels = np.array([line.strip().decode('utf-8').split('\t',1)[-1].split("\n")[0] for line in file])

		tokens = np.asarray(tokens).view(np.chararray)
		labels = np.asarray(labels).view(np.chararray)

		# split tokens on empty lines into sentences for position tracking
		# sentences = np.split(tokens,np.where(tokens=="\n")[0])

		# now remove all empty lines
		tokens = np.delete(tokens,np.where(tokens=="\n")[0])


		self.logger.info("Compiling set of prefixes and suffixes")

		# get comming prefixes
		prefixes = []
		for i in range(4):
			prefixes += list(set([x[0:i+1] for x in tokens]))


		# get comming suffixes
		suffixes = []
		for i in range(4):
			suffixes += list(set([x[len(x)-(i+1):] for x in tokens]))


		feats = [[]]

		self.logger.info("Extracting Token Features")

		for token in tokens:
			token = str(token)
			if token:
				curr = []
				if token in self.vocab:
					curr.append(self.vocab.index(token.lower()))

				for count, pre in enumerate(prefixes):
					if token.endswith(pre):
						curr.append(len(self.vocab)+count)
						break

				for count, suff in enumerate(suffixes):
					if token.endswith(suff):
						curr.append(len(self.vocab)+len(prefixes)+count)
						break


				if self.isnum(token) or self.text2int(token):
					curr.append(len(self.vocab)+len(prefixes)+len(suffixes))

				if token[0].isupper(): curr.append(len(self.vocab)+len(prefixes)+len(suffixes)+1)
				if token.isupper(): curr.append(len(self.vocab)+len(prefixes)+len(suffixes)+2)
				feats.append(curr)
			else:
				feats.append([])

		feats = np.asarray(feats[1:])

		self.logger.info("Finalizing Feature Extraction")

		self.logger.info("Features Shape: " + str(feats.shape))
		self.logger.info("Instnaces Shape: " + str(len(tokens)))
		self.logger.info("Labels Shape: " + str(len(labels)))

		return (feats, tokens, labels)

	def isnum(self,str):
		if str.isdigit(): return True
		if self.reg.match(str): return True
		return False


	def text2int(self,textnum, numwords={}):
		if not numwords:
			units = [
			"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
			"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
			"sixteen", "seventeen", "eighteen", "nineteen",
			]

			tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

			scales = ["hundred", "thousand", "million", "billion", "trillion"]

			numwords["and"] = (1, 0)
			for idx, word in enumerate(units):    numwords[word] = (1, idx)
			for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
			for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

		current = result = 0
		for word in textnum.split():
			if word not in numwords:
				return ""
			scale, increment = numwords[word]
			current = current * scale + increment
			if scale > 100:
				result += current
				current = 0

		return result + current


