import sys
import argparse
import logging
import pickle 
import os.path


from featureset import FeatureSet
from perceptron import Perceptron
from meta_data import MetaData


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


def train(results):
	# get files 

	vocab_file = results.vocab
	labels_file = results.labels
	train_file = results.train

	logger.debug(	'Started training with options:'		+ "\n" +
					'training file:	' + str(results.train) 	+ "\n" + 
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

	if not os.path.exists('model/train_feats'):
		train_feats = extract_feats(meta_data,train_file)
		logger.info("Writing extracted feats for training files to train.feats")
		write_obj(train_feats,'train.feats')
	else:
		logger.info("train_feats already exists ... loading.")
		train_feats = read_obj('train.feats')


	if not os.path.exists('model/model'):
		classifier = Perceptron(meta_data)
		classifier.train(train_feats)
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
		logger.info("Done feature extraction for testing data, writing in test.feats")
		test_feats = extract_feats(meta_data,test_file)
		write_obj(test_feats,'test.feats')
	else:
		logger.info("test.feats already exists ... loading.")
		test_feats = read_obj('test.feats')

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

if not os.path.exists('model'):
    os.makedirs('model')


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




