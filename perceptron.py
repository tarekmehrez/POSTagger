import numpy as np
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
        # self.feat_size = len(self.suffixes) + len(self.vocab) + 2
        self.feat_size = 4


    def train(self,feat_tuple):
        self._logger.info("Started training Perceptron")

        feat_idx = feat_tuple[0]
        inst_labels = feat_tuple[1]
        inst_vals = feat_tuple[2]

        self._theta = 0.1 * np.random.randn(len(self.labels),self.feat_size)


        for count, instance in enumerate(feat_idx):

            if not inst_vals[count]:
                continue
            else:

                self._logger.debug("Training instance: " + str(count) + ", out of: " + str(len(feat_idx)))

                compiled = self._compile_feats(instance)
                results = compiled[0]
                feats = compiled[1]
                label_idx = self.labels.index(inst_labels[count])

                # print feats
                for perc_count,pred in enumerate(results):
                    if perc_count == label_idx and pred < 1: self._theta[perc_count] = self._theta[perc_count] + feats[0]
                    if perc_count != label_idx and pred > 1: self._theta[perc_count] = self._theta[perc_count] - feats[0]


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
                compiled = self._compile_feats(instance)
                results = compiled[0]
                feat = compiled[1]

                f.write(inst_vals[count] + "\t" + self.labels[np.argmax(results)] + "\n")

        f.close()
        self._logger.info("Done Testing, results are written in pred.col")


    def get_theta(self):
        return self._theta


    def load_theta(self, theta):
        self._theta = theta

    def _compile_feats(self,instance):


        feat = np.asarray([instance])
        # feat = np.zeros([self.feat_size,1])

        # feat[instance[0]] = 1
        # feat[instance[1] + len(self.vocab)] = 1
        # feat[len(feat) - 2] = instance[2]
        # feat[len(feat) - 1] = instance[3]

        feats = np.repeat(feat.T,len(self.labels),axis=0)
        results = np.sum(feats * self._theta, axis =1)
        return (results,feats)