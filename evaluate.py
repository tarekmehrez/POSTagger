import numpy as np
import logging



class Evaluator(object):

	def __init__(self, labels_file, pred_file_path, gold_file_path):


		logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
		self._logger = logging.getLogger(__name__)
		self._logger.info("Reading tagged predictions and gold standard")

		with open(labels_file) as file:
			self.labels_set = [line.strip().decode('utf-8').split('\n',1)[0] for line in file]

		with open(pred_file_path) as file:
			self.instances = [line.strip().decode('utf-8').split('\t')[0] for line in file]

		with open(pred_file_path) as file:
			self.pred_labels = [line.strip().decode('utf-8').split('\t',1)[-1] for line in file]

		with open(gold_file_path) as file:
			self.gold_labels = [line.strip().decode('utf-8').split('\t',1)[-1] for line in file]





	def evaluate(self, output_file):

		self._logger.info("Starting evaluation")


		conf_matrix = np.zeros([len(self.labels_set),len(self.labels_set)])

		for row_count,should_be in enumerate(self.labels_set):
			gold_indices = [i for i, x in enumerate(self.gold_labels) if x == should_be]

			for predicted_as in gold_indices:

				predicted_label = self.pred_labels[predicted_as]
				# print predicted_label
				column_count = self.labels_set.index(predicted_label)
				conf_matrix[row_count, column_count] += 1



		f = open(output_file,'w')
		for count,i in enumerate(conf_matrix):
			f.write(str(self.labels_set[count]) + " ")
			for j in i:
				f.write(str(int(j)) + " ")
			f.write("\n")
		f.write("###########################\n")

		f.write("Class\tP\tR\tF\n")

		results = np.zeros([len(self.labels_set),3])

		for count,label in enumerate(self.labels_set):

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

		avg_p = np.sum(results,axis=0)[1] / len(self.labels_set)
		avg_r = np.sum(results,axis=0)[0] / len(self.labels_set)
		avg_f = np.sum(results,axis=0)[2] / len(self.labels_set)

		f.write("###########################\n")

		f.write("Precision: " + str("%.2f" % avg_p) + "\n")
		f.write("Recall:    " + str("%.2f" % avg_r) + "\n")
		f.write("f1 score:  " + str("%.2f" % avg_f) + "\n")
		f.write("Accuracy:  " + str("%.2f" % acc)   + "\n")


		f.close()
		self._logger.info("Done evaluation")



