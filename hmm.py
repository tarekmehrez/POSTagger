import numpy as np
import logging
import sys
import re
from collections import defaultdict

class HMM(object):

	def __init__(self):
		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)
		self.logger.info("Initializing HMM")

		self.num_reg = re.compile("^\d+((\,|\.|\/)\d+)*$")


	def train(self, train_file,freq_file):

		self.prepare_vocab(freq_file)
		self.logger.info("Started HMM Training")

		count = 0
		context = defaultdict(float)
		trans = defaultdict(float)
		emit = defaultdict(float)
		previous = '<s>'
		context[previous]+= 1

		for line in open(train_file):

			content = line.split('\t',1)

			if line != '\n':
				token = content[0]
				if token in self.vocab:
					token = str(self.replace_token(token,previous))
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

		self.theta = [context,trans,emit]

	def test(self,test_file,freq_file):
		self.prepare_vocab(freq_file)

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

				if token in self.vocab:
					token = str(self.replace_token(token,previous))

				curr = [1] * len(tags)


				for i,tag in enumerate(tags):
					emission = emit[tag+"_"+token]
					if not emission:
						emission = 10**-5
					curr[i] *= prev_prob * (trans[previous+"_"+tag]/context[previous]) * (emission/context[tag])


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

	def prepare_vocab(self,freq_file):

		with open(freq_file) as file:
			freq = [line.strip().decode('utf-8') for line in file]




		self.vocab = []
		for i in freq:
			content = i.strip().split(" ")
			if int(content[0]) <= 5:
				self.vocab.append(content[1])

		tokens = np.asarray(self.vocab).view(np.chararray)
		self.vocab = set(self.vocab)

		self.prefixes = []
		for i in range(3):
			self.prefixes += list(set([x[0:i+1] for x in tokens]))

		self.suffixes = []
		for i in range(3):
			self.suffixes += list(set([x[len(x)-(i+1):] for x in tokens]))
		self.logger.info("Tokens with freq less than 5: " + str(len(self.vocab)))
		self.logger.info("Common prefixes: " + str(len(self.prefixes)))
		self.logger.info("Common suffixes: " + str(len(self.suffixes)))

	def replace_token(self, token,previous):

		if token[0].isupper(): return "is_upper"
		if token.isupper(): return "is_capital"
		if len(token) > 1 and token.endswith('.'): return "is_abbrev"
		if previous  == '<s>': return "is_first"
		if self.isnum(token) or self.text2int(token): return "is_num"
		if re.match('^[\w-]+$', token) == None: return "is_special"

		for pre in reversed(self.prefixes):
			if token.startswith(pre):
				return "is_"+str(pre)

		for suff in reversed(self.suffixes):
			if token.endswith(suff):
				return "is_"+str(suff)

		if token.islower(): return "is_lower"

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

