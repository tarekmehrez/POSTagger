import numpy as np
import re
import logging

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
		
		self._reg = re.compile("^\d+((\,|\.)\d+)*$")
		
		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self._logger = logging.getLogger(__name__)

		self.vocab = meta_data[0]
		self.labels = meta_data[1]
		self.suffixes = meta_data[2]


	def extract_feats(self, file_path):

		self._logger.info("Started Feature Extraction")
		
		vocab_idx = []
		suff_idx = []
		num_idx = []


		inst_vals = []
		inst_labs = []

		f = open(file_path,'r')
		i=1
		for line in f.readlines():

			self._logger.debug("Reading data, line: " + str(i))
			i+= 1

			if line != "\n":
				content = line.split('\t',1)

				inst_vals.append(content[0])
				inst_labs.append(content[1].split("\n")[0])
				num_idx.append(self._isnum(content[0])*1)
				vocab_idx.append(self.vocab.index(content[0]))

		f.close()

		inst_vals = np.asarray(inst_vals).view(np.chararray)

		self._logger.info("Extracting Suffix Features")

		suff_idx = np.zeros(len(inst_vals))

		for count,suff in enumerate(self.suffixes):
			suff_idx += inst_vals.endswith(suff)*count

		self._logger.info("Finalizing Feature Extraction")

		feat_idx = np.zeros([len(inst_vals),4])

		feat_idx[:,0] = np.asarray(vocab_idx)
		feat_idx[:,1] = suff_idx
		feat_idx[:,2] = np.asarray(num_idx)
		feat_idx[:,3] = inst_vals.isupper()*1
		
		self._logger.info("Feature Extraction DONE, with feature size: " + str(feat_idx.shape))

		return (feat_idx, inst_labs)

	def _isnum(self,str):
		if str.isdigit(): return True
		if self._reg.match(str): return True
		return False
