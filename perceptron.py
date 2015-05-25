import numpy as np

feat_idx = np.loadtxt("out.txt")

vocab = 35934
classes_num = 49
suffixes = 13
capital = 1

total = vocab + suffixes + capital


theta = 0.1*np.random.randn(classes,total)

classes = []
instances = []


with open(fname) as f:
    instances.append(f.readlines().split(',',3))
    classes.append(label)
    


for instance in instances:

    feat = np.zeros(total)

	label = classes.index(instance[3])

    feat[instance[0]] = 1
    feat[instance[1] + vocab] = 1
    feat[len(feat) - 1] = instance[2]

    feats = np.repeat(feat,classes_num,axis=0)
    results = sum(feats * theta)
   
    for count,pred in enumerate(results):

    	if count == label and pred < 0:
    		theta[count] += feat
    	if count != label and pred > 0:
    		theta[count] -= feat

