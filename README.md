# POSTagger
Implementation of a POS Tagger using the Perceptron algorithm

The tool takes in a training file, vocabulary file & labels file to train a model, then a testing file should be passed to do the tagging.

# Steps: 


1- Run `prepare.sh` by passing the training file, to produce both the vocab and the labels files

```
./prepare.sh [training_file]
```


2- Run `main.py` and pass the training, vocab & labels files as follows:

```
python main.py --train [training_file] --vocab [vocab_file] --labels [labels_file]
```
This will produce several files in the `model` directory (if they are not already there), those files will be used to run the tagger on the testing file



3- Run `main.py` by passing the testing file as follows:

```
python main.py --test [testing_file]
```
This will produce the final tagged file in the `data` directory



###### An example on how to run the tool is availble in `run.sh ###### 


# Usage:

Options for running the tagger

```
usage: main.py [-h] [--train TRAIN] [--vocab VOCAB] [--labels LABELS]
               [--test TEST]

  -h, --help       show this help message and exit
  --train TRAIN    Training file
  --vocab VOCAB    Vocab file
  --labels LABELS  Labels file
  --test TEST      Testing file
  
```

