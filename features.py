import numpy as np
import re
import logging



'''
Features:
 
bag of words
common suffixes
is digit
is capital

'''
class Features(object):

	
	def __init__(self,file_path):
		self.file_path = file_path
		self.reg = re.compile("^\d+((\,|\.)\d+)*$")
		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self.logger = logging.getLogger(__name__)

		self.logger.info("Initializing Feature Set")

	def isnum(self,str):
		if str.isdigit():
			return True
		if self.reg.match(str):
			return True
		return False


	def extract_feats(self):

		self.logger.info("Started Feature Extraction")


		inst_vals = []
		self.inst_labs = []

		num_idx = []
		vocab = []
		vocab_idx = []

		f = open(self.file_path,'r')
		i=1
		for line in f.readlines():

			self.logger.debug("Reading data, line: + " + str(i))
			i+= 1

			if line == "\n":
				inst_vals.append(line)
				self.inst_labs.append(line)
				num_idx.append(0)
				vocab_idx.append(-1)
			else:
				content = line.split('\t',1)
				inst_vals.append(content[0])
				self.inst_labs.append(content[1].split("\n")[0])
				num_idx.append(self.isnum(content[0])*1)

				if content[0] not in vocab:
					vocab.append(content[0])
				
				vocab_idx.append(vocab.index(content[0]))


		f.close()


		inst_vals = np.asarray(inst_vals).view(np.chararray)
		self.labels = list(set(self.inst_labs))


		self.logger.info("Extracting Suffix Features")

		# extracting features

		feat_idx = np.zeros([len(inst_vals),5])

		suffixes = ["able", "al", "ed", "ing", "er", "est", "ion", "ive", "less", "ly", "ness", "ous", "." ]

		suff_idx = np.zeros(len(inst_vals))

		for count,suff in enumerate(suffixes):
			suff_idx += inst_vals.endswith(suff)*count

		self.logger.info("Finalizing Feature Extraction")


		feat_idx[:,0] = np.asarray(vocab_idx)
		feat_idx[:,1] = suff_idx
		feat_idx[:,2] = np.asarray(num_idx)
		feat_idx[:,3] = inst_vals.isupper()*1
		self.feat_idx = feat_idx.astype(int)
		np.savetxt('out.txt', feat_idx)

		self.logger.info("Feature Extraction DONE")


		def get_feats(self):
			return self.feat_idx

		def get_labels(self):
			return self.get_labels

		def get_gold_labels(self):
			return self.inst_labs
