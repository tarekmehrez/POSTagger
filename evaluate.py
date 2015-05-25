import numpy as np


with open("../data/pos/dev.col") as file:
	gold_file = np.array([line.strip().decode('utf-8').split('\t',1) for line in file])


with open("../data/pos/dev-predicted.col") as file:
	pred_file = np.array([line.strip().decode('utf-8').split('\t',1) for line in file])



if len(gold) != len(pred):
	print "Predictions and annotations have different sizes"


uniq_classes = []

for i in gold_file:
	if i != [u'']: uniq_classes.append(i[1].split("\n")[0])

uniq_classes = set(uniq_classes)

conf_matrix = numpy.zeros([len(uniq_classes) ,len(uniq_classes)])

