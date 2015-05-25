import sys

from features import Features

file_path = sys.argv[1]

feats = Features(file_path)
feats.extract_feats()