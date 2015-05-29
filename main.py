import sys
import argparse
import logging
import pickle 


from featureset import FeatureSet
from perceptron import Perceptron
from meta_data import MetaData


######## Org. functions ########

def help_exit():
	parser.print_help()
	sys.exit(1)


def write(obj, file_name):
	f = open(file_name, 'ab+')
	pickle.dump(f, obj)

def read(file_name):
	f = open(file_name, 'rb')
	return pickle.load(f)


######## Core functions ########


def train(results):
	# get files 

	vocab_file = results.vocab
	labels_file = results.labels
	train_file = results.train

	logger.debug(	'Started training with options:'		+ "\n" +
					'training file:	' + str(results.train) 	+ "\n" + 
					'vocab file:	' + str(results.vocab)	+ "\n" + 
					'labels file:	' + str(results.labels)	+ "\n")


	meta_data_instance = MetaData(vocab_file,labels_file)
	meta_data = meta_data_instance.get_meta_data()




	# extract features
	feature_set = FeatureSet(meta_data)

	# extract feats for training set
	train_feats = feature_set.extract_feats(train_file)

	# initialize classifier with meta data
	classifier = Perceptron(meta_data)

	# train classifier on training features
	classifier.train(train_feats)


	logger.info("Done Training, model is written in model file")
	model = classifier.get_theta()
	write(model, 'model')

	logger.info("Writing meta data file")
	write(meta_data,'meta_data')



def test(results):

	test_file = results.test


	logger.debug(	'Started testing with options:'			+ "\n" +
					'test file:	' + str(results.test) 		+ "\n")


	logger.info("Loading model and meta_data")
	model = read('model')
	meta_data= read('meta_data')


	feature_set = FeatureSet(meta_data)
	test_feats = feature_set.extract_feats(test_file)

	classifier = Perceptron(meta_data)
	classifier.load_theta(model)
	classifier.test(test_feats)




##############################################################################################################################

######## CMD args ########


parser = argparse.ArgumentParser()

parser.add_argument('--train', action='store', dest='train',
                    help='Input training file')

parser.add_argument('--vocab', action='store', dest='vocab',
                    help='Input training file')

parser.add_argument('--labels', action='store', dest='labels',
                    help='Input training file')


parser.add_argument('--test', action='store', dest='test',
                    help='Input training file')


results = parser.parse_args()


if len(sys.argv)==1:
	help_exit()

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)

if results.train and results.test:
	print "You can only do training or testing at a time"
	help_exit()

if results.train:
	if not results.vocab or not results.labels:
		print "You have to specify the vocab and labels file"
		help_exit()
	else:
		train(results)
		
if results.test:
	test(results)




