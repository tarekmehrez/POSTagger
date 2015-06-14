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
		
		self._reg = re.compile("^\d+((\,|\.)\d+)*$")
		
		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self._logger = logging.getLogger(__name__)

		self.vocab = meta_data[0]
		self.labels = meta_data[1]
		self.suffixes = meta_data[2]


	def extract_feats(self, file_path):

		self._logger.info("Started Feature Extraction")
		
		vocab_feats = [[0] * len(self.vocab)]
		suff_feats = []
		num_feats = []


		inst_vals = []
		inst_labs = []


		f = open(file_path,'r')
		i=1
		for line in f.readlines():

			# self._logger.debug("Extracting Features for line: " + str(i))
			i+= 1
			
			
			content = line.split('\t',1)
			if line == "\n":

				inst_vals.append('')
				inst_labs.append('')
				num_feats.append(-1)
				vocab_feats.append([-1] * len(self.vocab))
			else:
				token = str(content[0])
				label = str(content[1].split("\n")[0])


				inst_vals.append(token)
				inst_labs.append(label)
				num_feats.append(self._isnum(token)*1)

				current = [0] * len(self.vocab)
				current[self.vocab.index(token)] = 1

				vocab_feats.append(current)


		f.close()
		inst_vals = np.asarray(inst_vals).view(np.chararray)
		inst_labs = np.asarray(inst_labs).view(np.chararray)

		vocab_feats = np.asarray(vocab_feats[1:])

		self._logger.info("Extracting Suffix Features")

		suff_feats = np.zeros((len(inst_vals),len(self.suffixes)))

		for count,suff in enumerate(self.suffixes):

			has_suff = inst_vals.endswith(suff)*1
			suff_feats[has_suff,count] = 1


		self._logger.info("Finalizing Feature Extraction")


		total_feats = np.hstack((vocab_feats,suff_feats))

		num_feats = np.array([num_feats])
		cap_feats = np.array([inst_vals.isupper()*1])

		total_feats = np.hstack((total_feats,num_feats.T))
		total_feats = np.hstack((total_feats,cap_feats.T))

		feat_dense = csr_matrix(total_feats)


		self._logger.info("Features Shape: " + str(feat_dense.shape))
		self._logger.info("Instnaces Shape: " + str(len(inst_vals)))
		self._logger.info("Labels Shape: " + str(len(inst_labs)))

		print feat_dense.toarray()
		return (feat_dense, inst_labs, inst_vals)

	def _isnum(self,str):
		if str.isdigit(): return True
		if self._reg.match(str): return True
		return False
