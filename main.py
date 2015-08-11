import argparse
import logging
import pickle
import os.path
import sys

from featureset import FeatureSet
from perceptron_classifier import PerceptronClassifier
from evaluate import Evaluator
from hmm import HMM
######## Org. functions ########

def help_exit():
	parser.print_help()
	sys.exit(1)


def write_obj(obj, file_name):
	f = open('models/'+file_name, 'wb')
	pickle.dump(obj, f)
	f.close()

def read_obj(file_name):
	f = open('models/'+file_name, 'rb')
	return pickle.load(f)


######## Core functions ########

def extract_feats(input_file, output_file):
	feature_set = FeatureSet()
	if not os.path.exists('models/vocab'):
		feature_set.extract_feats(input_file,output_file,{})
	else:
		vocab = read_obj('vocab')
		feature_set.extract_feats(input_file,output_file,vocab)


def train(args):
	train_file = args.input
	if args.train_perceptron:

		classifier = PerceptronClassifier(args.input,args.feats,args.labels)
		classifier.train(float(args.step),int(args.iter))
		logger.info("Done Training, model is written in model file")
		model = classifier.get_theta()
		write_obj(model, 'model')

	elif args.train_hmm:
		classifier = HMM()
		classifier.train(args.input,args.freq)
		logger.info("Done Training, model is written in model file")
		model = classifier.get_theta()
		write_obj(model, 'hmm-model')


def test(args):
	if args.test_perceptron:

		model = read_obj('model')
		classifier = PerceptronClassifier(args.input,args.feats,args.labels)
		classifier.load_theta(model)
		classifier.test(args.output)

	elif args.test_hmm:

		logger.info("Loading model")
		model = read_obj('hmm-model')

		classifier = HMM()
		classifier.load_theta(model)
		classifier.test(args.input,args.freq)

def evaluate(args):

	evaluator = Evaluator(args.labels, args.pred,  args.gold)
	evaluator.evaluate(args.output)

##############################################################################################################################

######## CMD args ########


parser = argparse.ArgumentParser()

parser.add_argument('--extract', action='store', dest='extract',
                    help='Feature extraction')

parser.add_argument('--i', action='store', dest='input',
                    help='Input file')

parser.add_argument('--o', action='store', dest='output',
                    help='Output File')

parser.add_argument('--feats', action='store', dest='feats',
                    help='Features file')

parser.add_argument('--train-perceptron', action='store', dest='train_perceptron',
                    help='Training Perceptron Model')

parser.add_argument('--train-hmm', action='store', dest='train_hmm',
                    help='Training HMM Model')

parser.add_argument('--test-perceptron', action='store', dest='test_perceptron',
                    help='Testing Perceptron')

parser.add_argument('--test-hmm', action='store', dest='test_hmm',
                    help='Testing HMM')


parser.add_argument('--eval', action='store', dest='eval',
                    help='Evaluation. The following files must be specified: (Labels, Pred, Gold)')


parser.add_argument('--step', action='store', dest='step',
                    help='Decaying Step Size[Perceptron]')

parser.add_argument('--iter', action='store', dest='iter',
                    help='Training iterations[Perceptron]')

parser.add_argument('--labels', action='store', dest='labels',
                    help='Labels file - For Perceptron Training & Evaluation')

parser.add_argument('--freq', action='store', dest='freq',
                    help='Frequency file - For HMM Training')


parser.add_argument('--pred', action='store', dest='pred',
                    help='Evaluation: Tagged Predictions File')


parser.add_argument('--gold', action='store', dest='gold',
                    help='Evaluation: Gold Standard File')


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)


args = parser.parse_args()
print args
if len(sys.argv)==1:
	help_exit()

if not os.path.exists('models'):
	os.makedirs('models')

if args.extract:
	extract_feats(args.input, args.output)

elif args.train_perceptron or args.train_hmm:
	train(args)

elif args.test_perceptron or args.test_hmm:
	test(args)

elif args.eval:
	evaluate(args)