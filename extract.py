import numpy as np
import re

'''
Features:
 
bag of words
common suffixes
is digit
is capital

'''
reg = re.compile("^\d+((\,|\.)\d+)*$")

def isnum(str):
	if str.isdigit():
		return True
	if reg.match(str):
		return True
	return False


# reading data
print "reading data"

inst_vals = []
inst_labs = []

num_idx = []
vocab = []
vocab_idx = []

f = open('../data/pos/train.col','r')

i = 1

for line in f.readlines():
	print i 
	i += 1
	if line == "\n":
		inst_vals.append(line)
		inst_labs.append(line)
		num_idx.append(0)
		vocab_idx.append(-1)
	else:

		# convert strings to numbers, remove floating points and commas in numbers
		content = line.split('\t',1)
		inst_vals.append(content[0])
		inst_labs.append(content[1].split("\n")[0])
		num_idx.append(isnum(content[0])*1)

		if content[0] not in vocab:
			vocab.append(content[0])
		
		vocab_idx.append(vocab.index(content[0]))


f.close()


inst_vals = np.asarray(inst_vals).view(np.chararray)
labels = list(set(inst_labs))



# extracting features
print "extracting features"

feat_idx = np.zeros([len(inst_vals),4])

suffixes = ["able", "al", "ed", "ing", "er", "est", "ion", "ive", "less", "ly", "ness", "ous", "." ]

suff_idx = np.zeros(len(inst_vals))

for count,suff in enumerate(suffixes):
	suff_idx += inst_vals.endswith(suff)*count

feat_idx[:,0] = np.asarray(vocab_idx)
feat_idx[:,1] = suff_idx
feat_idx[:,2] = np.asarray(num_idx)
feat_idx[:,3] = inst_vals.isupper()*1
feat_idx = feat_idx.astype(int)
np.savetxt('out.txt', feat_idx)

