import logging
import numpy as np
import sys
from perceptron import Perceptron

class PerceptronClassifier(object):


    def __init__(self,input_file,feats,labels_file):

        logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
        self.logger = logging.getLogger(__name__)

        self.logger.info("Initializing Perceptron")
        with open(labels_file) as file:
            self.labels_set = [line.strip().decode('utf-8').split('\n',1)[0] for line in file]

        self.feats = feats

        with open(input_file) as file:
            self.tokens = [line.strip().decode('utf-8').split('\t',1)[0] for line in file]

        with open(input_file) as file:
            self.tags = [line.strip().decode('utf-8').split('\t',1)[-1].split("\n")[0] for line in file]



    def train(self,step,iterations):
        self.logger.info("Started training Perceptron")



        self.perceptrons = []
        for i in range(len(self.labels_set)):
            self.perceptrons.append(Perceptron())

        for iteration in range(iterations):

            correct = 0
            wrong = 0
            self.logger.info("Training Iteration: "+ str(iteration + 1))

            with open(self.feats) as f:
                for count,instance in enumerate(f):
                    instance = instance.split('\n',1)[0]
                    if instance:
                        current_feats = eval(instance)
                        gold_label = self.tags[count]
                        results = []

                        for perceptron in self.perceptrons:
                            results.append(perceptron.activate(current_feats,0))

                        if results[self.labels_set.index(gold_label)] ==1:
                            correct += 1
                        else:
                            wrong += 1

                        for idx, result in enumerate(results):
                            if self.labels_set[idx] == gold_label and result == -1: self.perceptrons[idx].inc_weight(current_feats,step)
                            if self.labels_set[idx] != gold_label and result == 1: self.perceptrons[idx].dec_weight(current_feats,step)

            self.logger.info("Correct/Incorrect Classifications: "+ str(correct) + "/" + str(wrong))
            step -= step * 0.1

    def test(self,output):
        self.logger.info("Started testing Perceptron")

        out_file = open(output,'w')

        with open(self.feats) as f:
            for count,instance in enumerate(f):
                testing_instance = instance.split('\n',1)[0]

                if not testing_instance:
                    out_file.write('\n')
                    continue
                else:
                    results = []
                    current_feats = eval(testing_instance)

                    for perceptron in self.perceptrons:
                        results.append(perceptron.activate(current_feats,1))

                    winner = results.index(max(results))

                    out_file.write(self.tokens[count] + "\t" + self.labels_set[winner] + "\n")

        out_file.close()
        self.logger.info("Done Testing, results are written in pred.col")


    def get_theta(self):
        return self.perceptrons


    def load_theta(self, perceptrons):
        self.perceptrons = perceptrons
