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
		
		feats = [[]]

		inst_vals = []
		inst_labs = []


		f = open(file_path,'r')
		for line in f.readlines():

			
			content = line.split('\t',1)
			if line == "\n":
				feats.append([])
				inst_vals.append('')
				inst_labs.append('')

			else:
				token = str(content[0])
				label = str(content[1].split("\n")[0])


				inst_vals.append(token)
				inst_labs.append(label)
				curr = [self.vocab.index(token)]

				for count, suff in enumerate(self.suffixes):
					if token.endswith(suff):
						curr.append(len(self.vocab)+count)
						break

				if self._isnum(token): curr.append(len(self.vocab)+len(self.suffixes))
				if token[0].isupper(): curr.append(len(self.vocab)+len(self.suffixes)+1)
				# feats.append([self.vocab.index(token),self._isnum(token)*1,token[0].isupper()*1])
				feats.append(curr)

		f.close()
		inst_vals = np.asarray(inst_vals).view(np.chararray)
		inst_labs = np.asarray(inst_labs).view(np.chararray)
		feats = np.asarray(feats[1:])

		self._logger.info("Finalizing Feature Extraction")

		self._logger.info("Features Shape: " + str(feats.shape))
		self._logger.info("Instnaces Shape: " + str(len(inst_vals)))
		self._logger.info("Labels Shape: " + str(len(inst_labs)))

		return (feats, inst_labs, inst_vals)

	def _isnum(self,str):
		if str.isdigit(): return True
		if self._reg.match(str): return True
		return False
