import numpy as np
import logging



class Evaluator(object):

	def __init__(self, meta_data, pred_file_path, gold_file_path):


		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self._logger = logging.getLogger(__name__)
		self._logger.info("Reading tagged predictions and gold standard")

		self.vocab = meta_data[0]
		self.labels = meta_data[1]

		self.instances = []
		self.pred_labels = []
		self.gold_labels = []

		with open(pred_file_path) as file:
			self.instances = [line.strip().decode('utf-8').split('\t')[0] for line in file]

		with open(pred_file_path) as file:
			self.pred_labels = [line.strip().decode('utf-8').split('\t',1)[-1] for line in file]

		with open(gold_file_path) as file:
			self.gold_labels = [line.strip().decode('utf-8').split('\t',1)[-1] for line in file]





	def evaluate(self):

		self._logger.info("Starting evaluation")
		conf_matrix = np.zeros([len(self.labels),len(self.labels)])


		for label in self.labels:
			if label not in self.gold_labels:
				self.labels.remove(label)

		for row_count,should_be in enumerate(self.labels):
			gold_indices = [i for i, x in enumerate(self.gold_labels) if x == should_be]

			for predicted_as in gold_indices:

				predicted_label = self.pred_labels[predicted_as]

				if predicted_label not in self.labels:
					conf_matrix = np.resize(conf_matrix,(conf_matrix.shape[0]+1,conf_matrix.shape[1]+1))
					self.labels.append(predicted_label)

				column_count = self.labels.index(predicted_label)



				conf_matrix[row_count, column_count] += 1



		f = open('eval.out','w')
		for i in conf_matrix:
			for j in i:
				f.write(str(int(j)) + " ")
			f.write("\n")
		f.write("###########################\n")

		f.write("Class\tP\tR\tF\n")

		results = np.zeros([len(self.labels),3])

		for count,label in enumerate(self.labels):

			f.write(label + "\t")

			if np.sum(conf_matrix, axis=0)[count] == 0: precision = 0
			else: precision = conf_matrix[count][count] / np.sum(conf_matrix, axis=0)[count]
			f.write(str("%.2f" % precision) + "\t")

			if  np.sum(conf_matrix, axis=1)[count] == 0: recall = 0
			else :recall = conf_matrix[count][count] / np.sum(conf_matrix, axis=1)[count]
			f.write(str("%.2f" % recall) + "\t")

			if  (precision + recall) == 0:  fscore = 0
			else: fscore = (2 * precision * recall) / (precision + recall)
			f.write(str("%.2f" % fscore)+ "\n")

			results[count] = [precision,recall,fscore]


		acc = np.trace(conf_matrix) / np.sum(conf_matrix)

		avg_p = np.sum(results,axis=0)[1] / len(self.labels)
		avg_r = np.sum(results,axis=0)[0] / len(self.labels)
		avg_f = np.sum(results,axis=0)[2] / len(self.labels)

		f.write("###########################\n")

		f.write("Precision: " + str("%.2f" % avg_p) + "\n")
		f.write("Recall:    " + str("%.2f" % avg_r) + "\n")
		f.write("f1 score:  " + str("%.2f" % avg_f) + "\n")
		f.write("Accuracy:  " + str("%.2f" % acc)   + "\n")


		f.close()


