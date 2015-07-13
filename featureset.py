import numpy as np
from scipy.sparse import csr_matrix

import re
import logging
import sys
import time
from collections import defaultdict
import operator


'''
Features:
vocab id
common suffix id
is digit
is capital

'''
class FeatureSet(object):


	def __init__(self, meta_data):

		self.num_reg = re.compile("^\d+((\,|\.|\/)\d+)*$")
		self.char_reg = re.compile("^(\,|\.|\:|\;|\!|\#|\$|\%|\&|\*|\(|\)|\{|\[|\]|\}|\?|@|\'\'|\'|\"|\`|\\\)+$")
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
			tokens = [line.strip().decode('utf-8').split('\t',1)[0] for line in file]

		with open(file_path) as file:
			tags = [line.strip().decode('utf-8').split('\t',1)[-1].split("\n")[0] for line in file]

		tokens = np.asarray(tokens).view(np.chararray)
		tags = np.asarray(tags).view(np.chararray)

		# get length of longest sentence (for feature extraction)
		breaks = np.where(tokens=="")[0]+1
		breaks = np.insert(breaks,0,0)
		diff = [x - breaks[i - 1] for i, x in enumerate(breaks)][1:]
		max_sent = np.max(diff)

		# now remove all empty lines
		tokens = np.delete(tokens,np.where(tokens=="\n")[0])


		self.logger.info("Compiling set of prefixes and suffixes")



		prefixes = defaultdict(float)
		suffixes = defaultdict(float)

		# get common prefixes
		for i in [2,3,4,5]:
			for x in tokens:
				prefixes[str(x[0:i])] += 1

		prefixes = sorted(prefixes.items(), key=operator.itemgetter(1))[-200:]
		self.logger.info("Prefixes size: " + str(len(prefixes)))

		# get common suffixes
		for i in [2,3,4,5]:
			for x in tokens:
				suffixes[x[len(x)-(i):]] += 1

		suffixes = sorted(suffixes.items(), key=operator.itemgetter(1))[-200:]


		self.logger.info("Suffixes size: " + str(len(suffixes)))
		self.logger.info("Vocab size: " + str(len(self.vocab)))

		feats = []

		self.logger.info("Extracting Token Features")
		position = -1
		max_length = len(max(tokens, key=len))
		for count,token in enumerate(tokens):
			token = str(token)

			if token:
				curr_dim = 0
				position+=1
				curr = []

				# bias value
				curr.append(0)
				curr_dim += 1

				# token form index
				if count > 0 and tokens[count-1] in self.vocab: curr.append(self.vocab.index(tokens[count-1].lower()))
				curr_dim += len(self.vocab)

				if count > 1 and tokens[count-2] in self.vocab: curr.append(self.vocab.index(tokens[count-2].lower()))
				curr_dim += len(self.vocab)


				if token in self.vocab: curr.append(curr_dim + self.vocab.index(token.lower()))
				curr_dim += len(self.vocab)


				if count < len(tokens)-1 and tokens[count+1] in self.vocab: curr.append(curr_dim + self.vocab.index(tokens[count+1].lower()))
				curr_dim += len(self.vocab)

				if count < len(tokens)-2 and tokens[count+2] in self.vocab: curr.append(curr_dim + self.vocab.index(tokens[count+1].lower()))
				curr_dim += len(self.vocab)

				# prefix index
				for count, pre in enumerate(prefixes):
					if token.startswith(pre[0]):
						curr.append(curr_dim+count)
						break
				curr_dim += len(prefixes)
				# suffix index
				for count, suff in enumerate(suffixes):
					if token.endswith(suff[0]):
						curr.append(curr_dim+count)
						break
				curr_dim += len(suffixes)

				# is number
				if self.isnum(token) or self.text2int(token):
					curr.append(curr_dim)
				curr_dim += 1

				# is upper
				if token[0].isupper(): curr.append(curr_dim)
				curr_dim += 1

				# is capitalized
				if token.isupper(): curr.append(curr_dim)
				curr_dim += 1

				# is abbreviated
				if len(token) > 1 and token.endswith('.'): curr.append(curr_dim)
				curr_dim += 1

				# is special character
				if self.char_reg.match(token): curr.append(curr_dim)
				curr_dim += 1

				# contains hyphen
				if '-' in token: curr.append(curr_dim)
				curr_dim += 1

				# contains digit
				if self.contains_digits(token):  curr.append(curr_dim)
				curr_dim += 1

				# length features
				curr.append(curr_dim+len(token))
				curr_dim += max_length

				# position features
				curr.append(curr_dim+position)
				curr_dim += max_sent

				# done writing feature indices
				feats.append(curr)

			# no token = "\n" -> sentence ends, new sentence
			else:
				feats.append([])
				position = -1

		feats = np.asarray(feats)

		self.logger.info("Finalizing Feature Extraction")

		self.logger.info("Feature Dimensions: " + str(curr_dim))
		self.logger.info("Instnaces Number: " + str(len(tokens)))
		self.logger.info("Labels Number: " + str(len(tags)))

		return (feats, tokens, tags, curr_dim)

	def contains_digits(self,d):
		_digits = re.compile('\d')
		return bool(_digits.search(d))

	def isnum(self,str):
		if str.isdigit(): return True
		if self.num_reg.match(str): return True
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


