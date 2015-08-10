import numpy as np
import logging

from scipy.sparse import csr_matrix
from perceptron import Perceptron

class PerceptronClassifier(object):


    def __init__(self, meta_data):

        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing Perceptron")
        self.vocab = meta_data[0]
        self.labels = meta_data[1]


    def train(self,feat_tuple,step,iterations):
        self.logger.info("Started training Perceptron")

        feats = feat_tuple[0]
        tokens = feat_tuple[1]
        tags = feat_tuple[2]

        self.perceptrons = []
        for i in range(len(self.labels)):
            self.perceptrons.append(Perceptron())

        for iteration in range(iterations):

            correct = 0
            wrong = 0
            self.logger.info("Training Iteration: "+ str(iteration + 1))

            for count,training_instance in enumerate(feats):

                if training_instance:
                    current_feats = training_instance.get_feats()
                    gold_label = tags[count]
                    results = []
                    for perceptron in self.perceptrons:
                        results.append(perceptron.activate(current_feats))

                    if results[self.labels.index(gold_label)] ==1:
                        correct += 1
                    else:
                        wrong += 1
                    for idx, result in enumerate(results):
                        if self.labels[idx] == gold_label and result == -1: self.perceptrons[idx].inc_weight(current_feats,step)
                        if self.labels[idx] != gold_label and result == 1: self.perceptrons[idx].dec_weight(current_feats,step)

            self.logger.info("Correct/Incorrect Classifications: "+ str(correct) + "/" + str(wrong))
            step -= step * 0.1

    def test(self,feat_tuple):
        self.logger.info("Started testing Perceptron")

        feats = feat_tuple[0]
        tokens = feat_tuple[1]
        tags = feat_tuple[2]

        f = open('data/pred.col','w')

        for count, testing_instance in enumerate(feats):


            if not testing_instance:
                f.write('\n')
                continue
            else:
                results = []
                current_feats = testing_instance.get_feats()

                for perceptron in self.perceptrons:
                    results.append(perceptron.activate(current_feats))

                winner = results.index(max(results))

                f.write(tokens[count] + "\t" + self.labels[winner] + "\n")

        f.close()
        self.logger.info("Done Testing, results are written in pred.col")


    def get_theta(self):
        return self.perceptrons


    def load_theta(self, perceptrons ):
        self.perceptrons = perceptrons
