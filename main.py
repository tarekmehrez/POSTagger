import sys
import argparse
import logging
import pickle
import os.path


from featureset import FeatureSet
from perceptron import Perceptron
from evaluate import Evaluator
from meta_data import MetaData
from hmm import HMM
######## Org. functions ########

def help_exit():
	parser.print_help()
	sys.exit(1)


def write_obj(obj, file_name):
	f = open('model/'+file_name, 'wb')
	pickle.dump(obj, f)
	f.close()

def read_obj(file_name):
	f = open('model/'+file_name, 'rb')
	return pickle.load(f)


######## Core functions ########

def extract_feats(meta_data,feat_file):
	feature_set = FeatureSet(meta_data)
	return feature_set.extract_feats(feat_file)

def hmm_train(results):
	train_file = results.train
	freq_file = results.freq

	logger.debug(	'Started training HMM with options:'	+ "\n" +
					'training file:	' + str(train_file) 	+ "\n" +
					'frequency file:' + str(freq_file)		+ "\n")


	if not os.path.exists('model/hmm-model'):
		classifier = HMM()
		classifier.train(train_file,freq_file)
		logger.info("Done Training, model is written in model file")
		model = classifier.get_theta()
		write_obj(model, 'hmm-model')
	else:
		logger.info('model already exists, nothing to do!')

def hmm_test(results):

	test_file = results.test
	freq_file = results.freq


	logger.debug(	'Started testing HMM with options:'		+ "\n" +
					'test file:	'		+ str(results.test) + "\n" +
					'frequency file:' 	+ str(freq_file)	+ "\n")

	logger.info("Loading model")
	model = read_obj('hmm-model')


	classifier = HMM()

	classifier.load_theta(model)
	classifier.test(test_file,freq_file)

def train(results):
	# get files

	vocab_file = results.vocab
	labels_file = results.labels
	train_file = results.train
	step = results.step
	iterations = results.iter


	if not step:
		step = 0.01
	if not iterations:
		iterations = 10

	logger.debug(	'Started training with options:'		+ "\n" +
					'training file:	' + str(train_file) 	+ "\n" +
					'step size: '	  + str(step)			+ "\n" +
					'no. of iter: '	  + str(iterations)		+ "\n" +
					'vocab file:	' + str(vocab_file)		+ "\n" +
					'labels file:	' + str(labels_file)	+ "\n")




	if not os.path.exists('model/meta_data'):
		meta_data_instance = MetaData(vocab_file,labels_file)
		meta_data = meta_data_instance.get_meta_data()
		logger.info("Writing meta data file")
		write_obj(meta_data,'meta_data')
	else:
		logger.info("meta data file already exists ... loading")
		meta_data = read_obj('meta_data')

	if not os.path.exists('model/train.feats'):
		train_feats = extract_feats(meta_data,train_file)
		logger.info("Writing extracted feats for training files to train.feats")
		write_obj(train_feats,'train.feats')
	else:
		logger.info("train.feats already exists ... loading.")
		train_feats = read_obj('train.feats')

	if not os.path.exists('model/model'):
		classifier = Perceptron(meta_data)
		classifier.train(train_feats,step,iterations)
		logger.info("Done Training, model is written in model file")
		model = classifier.get_theta()
		write_obj(model, 'model')
	else:
		logger.info('model already exists, nothing to do!')


def test(results):

	test_file = results.test


	logger.debug(	'Started testing with options:'			+ "\n" +
					'test file:	' + str(results.test) 		+ "\n")


	logger.info("Loading model and meta_data")
	model = read_obj('model')
	meta_data= read_obj('meta_data')

	if not os.path.exists('model/test.feats'):
		test_feats = extract_feats(meta_data,test_file)
		logger.info("Done feature extraction for testing data, writing in test.feats")
		write_obj(test_feats,'test.feats')
	else:
		logger.info("test.feats already exists ... loading.")
		test_feats = read_obj('test.feats')

	classifier = Perceptron(meta_data)

	classifier.load_theta(model)
	classifier.test(test_feats)

def evaluate(results):
	vocab_file = results.vocab
	labels_file = results.labels

	gold_file = results.gold
	pred_file = results.pred

	logger.debug(	'Started evaluation with options:'		+ "\n" +
					'gold file:		' + str(results.gold) 	+ "\n" +
					'pred file:		' + str(results.pred) 	+ "\n" +
					'vocab file:	' + str(results.vocab)	+ "\n" +
					'labels file:	' + str(results.labels)	+ "\n")


	if not os.path.exists('model/meta_data'):
		meta_data_instance = MetaData(vocab_file,labels_file)
		meta_data = meta_data_instance.get_meta_data()
		logger.info("Writing meta data file")
		write_obj(meta_data,'meta_data')
	else:
		logger.info("meta data file already exists ... loading")
		meta_data = read_obj('meta_data')


	evaluator = Evaluator(meta_data, pred_file, gold_file)
	evaluator.evaluate()

##############################################################################################################################

######## CMD args ########


parser = argparse.ArgumentParser()

parser.add_argument('--class', action='store', dest='classifier',
                    help='0: Perceptron, 1: HMM')

parser.add_argument('--train', action='store', dest='train',
                    help='Training file')

parser.add_argument('--step', action='store', dest='step',
                    help='Decaying Step Size')

parser.add_argument('--iter', action='store', dest='iter',
                    help='Training iterations')

parser.add_argument('--vocab', action='store', dest='vocab',
                    help='Vocab file - For Perceptron Training & Evaluation')

parser.add_argument('--labels', action='store', dest='labels',
                    help='Labels file - For Perceptron Training & Evaluation')


parser.add_argument('--freq', action='store', dest='freq',
                    help='Frequency file - For HMM Training')


parser.add_argument('--test', action='store', dest='test',
                    help='Testing file')

parser.add_argument('--eval', action='store', dest='eval',type=int, default=0,choices=[0, 1],
                    help='Set to 1, to run evaluation [default=0]. The following files must be specified: (Vocab, Labels, Pred, Gold)')

parser.add_argument('--pred', action='store', dest='pred',
                    help='In case eval was set to 1: Tagged Predictions File')


parser.add_argument('--gold', action='store', dest='gold',
                    help='In case eval was set to 1: Gold Standard File')


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)


results = parser.parse_args()
if len(sys.argv)==1:
	help_exit()

if results.train and results.test:
	print "You can only do training or testing at a time"
	help_exit()


# train

if not os.path.exists('model'):
	os.makedirs('model')

if results.eval:
	if not (results.pred and results.gold and results.vocab and results.labels):
		print "Predictions, gold annotations, vocab & labels files must be specified when eval is set to 1"
		help_exit()
	else:
		evaluate(results)

elif int(results.classifier) not in [0,1]:
	print "Possible values for --class are 0 or 1"
	help_exit()


if results.train:

	if int(results.classifier) == 1:
		hmm_train(results)

	else:
		if not results.vocab or not results.labels:
			print "You have to specify the vocab and labels file"
			help_exit()
		elif results.step <1:
			print "Step size should be greater than 1"
			help_exit()
		elif results.iter <1:
			print "Number of iterations should be greater than 1"
			help_exit()
		else:
			train(results)
# test

elif results.test:
	if int(results.classifier) == 1:
		hmm_test(results)
	else:
		test(results)

# eval







