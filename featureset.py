import numpy as np
import re
import logging
import sys
import time
import operator

from scipy.sparse import csr_matrix
from collections import defaultdict
from token import Token


class FeatureSet(object):


	def __init__(self, meta_data):

		self.num_reg = re.compile("^\d+((\,|\.|\/)\d+)*$")
		self.char_reg = re.compile("^(\,|\.|\:|\;|\!|\#|\$|\%|\&|\*|\(|\)|\{|\[|\]|\}|\?|@|\'\'|\'|\"|\`|\\\)+$")

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')

		self.logger = logging.getLogger(__name__)

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

		# now remove all empty lines
		tokens = np.delete(tokens,np.where(tokens=="\n")[0])
		tags = np.delete(tags,np.where(tokens=="\n")[0])

		feats = []

		for count,token in enumerate(tokens):
			token = str(token)

			if token:

				# token form
				curr_token = Token(token)

				# prev token form
				if tokens[count-1]: curr_token.set_feat("PREVIOUS_FORM",str(tokens[count-1]))

				# next token form
				if tokens[count+1]: curr_token.set_feat("NEXT_FORM",str(tokens[count+1]))


				# is number
				if self.isnum(token) or self.text2int(token): curr_token.set_feat("IS_NUM",1)


				# is upper
				if token[0].isupper(): curr_token.set_feat("IS_UPP",1)

				# is capitalized
				if token.isupper(): curr_token.set_feat("IS_CAP",1)


				# is abbreviated
				if len(token) > 1 and token.endswith('.'): curr_token.set_feat("IS_ABB",1)

				# is special character
				if self.char_reg.match(token): curr_token.set_feat("IS_CHAR",1)

				# contains hyphen
				if '-' in token: curr_token.set_feat("HAS_HYPH",1)

				# contains digit
				if self.contains_digits(token): curr_token.set_feat("HAS_DIG",1)

				feats.append(curr_token)


			# no token = "\n" -> sentence ends, new sentence
			else:
				feats.append([])

		self.logger.info("Finalizing Feature Extraction")

		self.logger.info("Instnaces Number: " + str(len(tokens)))
		self.logger.info("Labels Number: " + str(len(tags)))

		return (feats, tokens, tags)
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


