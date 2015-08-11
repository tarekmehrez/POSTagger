import re
import logging
import pickle

class FeatureSet(object):


	def __init__(self):

		self.num_reg = re.compile("^\d+((\,|\.|\/)\d+)*$")
		self.char_reg = re.compile("^(\,|\.|\:|\;|\!|\#|\$|\%|\&|\*|\(|\)|\{|\[|\]|\}|\?|@|\'\'|\'|\"|\`|\\\)+$")

		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')

		self.logger = logging.getLogger(__name__)

	def extract_feats(self, input_file,output_file,vocab):

		self.logger.info("Started Feature Extraction")

		with open(input_file) as file:
			tokens = [line.strip().decode('utf-8').split('\t',1)[0] for line in file]


		file = open(output_file, 'wb')
		training = vocab == []

		for count,token in enumerate(tokens):
			token = str(token)

			if token:
				curr_feats = []

				# token form
				if training and token not in vocab:
					vocab.append(token.lower())
				curr_feats.append("FORM_"+str(vocab.index(token.lower())))

				# prev token form
				if tokens[count-1]:
					if training and tokens[count-1] not in vocab:
						vocab.append(tokens[count-1].lower())
					curr_feats.append("PREV_FORM_"+str(vocab.index(tokens[count-1].lower())))

				else:
					curr_feats.append("IS_FIRST")

				# next token form
				if tokens[count+1]:
					if training and tokens[count+1] not in vocab:
						vocab.append(tokens[count+1].lower())
					curr_feats.append("NEXT_FORM_"+str(vocab.index(tokens[count+1].lower())))
				else:
					curr_feats.append("IS_LAST")

				# is number
				if self.isnum(token): curr_feats.append("IS_NUM")

				# is upper
				if token[0].isupper(): curr_feats.append("IS_UPP")

				# is capitalized
				if token.isupper(): curr_feats.append("IS_CAP")

				# is abbreviated
				if len(token) > 1 and token.endswith('.'): curr_feats.append("IS_ABB")

				# is special character
				if self.char_reg.match(token): curr_feats.append("IS_CHAR")

				file.write(str(curr_feats) +"\n")

			# no token = "\n" -> sentence ends, new sentence
			else:
				file.write("\n")

		self.logger.info("Finalizing Feature Extraction")

		if training:
			f = open('models/vocab', 'wb')
			pickle.dump(vocab, f)
			f.close()


		file.close()

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