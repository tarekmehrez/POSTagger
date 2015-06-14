import numpy as np
from scipy.sparse import csr_matrix

import logging
import pickle
import sys
class Perceptron(object):


    def __init__(self, meta_data):

        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
        self._logger = logging.getLogger(__name__)

        self._logger.info("Initializing Perceptron")


        self._theta = []
        self.meta_data = meta_data

        self.vocab = meta_data[0]
        self.labels = meta_data[1]
        self.suffixes = meta_data[2]
        self.feat_size = len(self.suffixes) + len(self.vocab) + 2
        # self.feat_size = 4


    def train(self,feat_tuple):
        self._logger.info("Started training Perceptron")


        feat_idx = feat_tuple[0]
        inst_labels = np.asarray(feat_tuple[1]).view(np.chararray)
        inst_vals = feat_tuple[2]

        self._theta = 0.1 * np.random.randn(len(self.labels),self.feat_size)


        for count, instance in enumerate(feat_idx):

            if not inst_vals[count]:
                continue
            else:

                # self._logger.debug("Training instance: " + str(count) + ", out of: " + str(feat_idx.shape[0]))

                
                results = instance.multiply(self._theta)
                results = results.sum(axis=1)

                label_idx = self.labels.index(inst_labels[count])

                if results[label_idx] < 1:
                    self._theta[label_idx] = self._theta[label_idx] + instance


                indices = np.asarray(np.where(results > 1)[0]).ravel().tolist()

                if label_idx in indices:                
                    indices = indices.remove(label_idx)

                if indices:
                    self._theta[indices] =  self._theta[indices]  - instance.toarray()


    def test(self,feat_tuple):
        self._logger.info("Started testing Perceptron")

        feat_idx = feat_tuple[0]
        inst_labels = feat_tuple[1]
        inst_vals = feat_tuple[2]
        
        f = open('data/pred.col','w')

        for count, instance in enumerate(feat_idx):
            self._logger.debug("Testing instance: " + str(count))


            if not inst_vals[count]:                
                f.write('\n')
                continue
            else:
                results = instance.multiply(self._theta)
                results = results.sum(axis=1)
 
                # print results

                f.write(inst_vals[count] + "\t" + self.labels[np.argmax(results)] + "\n")

        f.close()
        self._logger.info("Done Testing, results are written in pred.col")


    def get_theta(self):
        return self._theta


    def load_theta(self, theta):
        self._theta = theta
