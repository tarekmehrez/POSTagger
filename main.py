import sys

from featureset import FeatureSet
from perceptron import Perceptron


# get files 

vocab_file = sys.argv[1]
labels_file = sys.argv[2]
train_file = sys.argv[3]
# test_file = sys.argv[4]



feature_set = FeatureSet(vocab_file, labels_file)
meta_data = feature_set.get_meta_data()

# extract feats for training set
train_feats = feature_set.extract_feats(train_file)

# initialize classifier with meta data
classifier = Perceptron(meta_data)

# train classifier on training features
classifier.train(train_feats)

# extract feats for testing set
# test_feats = feature_set.extract_feats(test_file)

# test model
# classifier.test(test_feats)