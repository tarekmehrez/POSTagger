import numpy as np
from scipy.sparse import csr_matrix

import logging
import pickle
import sys
class Perceptron(object):


    def __init__(self, meta_data):

        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing Perceptron")


        self.theta = []
        self.meta_data = meta_data

        self.vocab = meta_data[0]
        self.labels = meta_data[1]


    def train(self,feat_tuple):
        self.logger.info("Started training Perceptron")


        feat_idx = feat_tuple[0]
        inst_vals = feat_tuple[1]
        inst_labels = feat_tuple[2]
        feat_size = feat_tuple[3]

        self.theta = 0.001 * np.random.randn(len(self.labels),feat_size)
        step_size = 0.01
        for i in range(10):
            self.logger.info("training iteration: " + str(i))
            weights_copy = self.theta
            for count, instance in enumerate(feat_idx):

                if not instance:
                    continue
                else:
                    np.random.shuffle(instance)
                    results = np.sum(weights_copy[:,instance],axis=1)

                    label_idx = self.labels.index(inst_labels[count])

                    for perc_count,pred in enumerate(results):
                        if perc_count == label_idx and pred < 1: weights_copy[perc_count][instance] += 1 * step_size
                        if perc_count != label_idx and pred > 1: weights_copy[perc_count][instance] -= 1 * step_size
            self.theta = weights_copy
            step_size -= 0.001

    def test(self,feat_tuple):
        self.logger.info("Started testing Perceptron")

        feat_idx = feat_tuple[0]
        inst_vals = feat_tuple[1]
        inst_labels = feat_tuple[2]

        f = open('data/pred.col','w')

        for count, instance in enumerate(feat_idx):


            if not instance:
                f.write('\n')
                continue
            else:
                results = np.sum(self.theta[:,instance],axis=1)


                f.write(inst_vals[count] + "\t" + self.labels[np.argmax(results)] + "\n")

        f.close()
        self.logger.info("Done Testing, results are written in pred.col")


    def get_theta(self):
        return self.theta


    def load_theta(self, theta):
        self.theta = theta
