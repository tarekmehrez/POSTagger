# POSTagger
Implementation of a POS Tagger using Perceptron and HMMs


The tool takes in a training file, vocabulary file & labels file to train a model, then a testing file should be passed to do the tagging.


Before any training steps, run `prepare.sh` by passing the training file, to produce vocab, labels and freq files

```
./prepare.sh [training_file]
```


# Steps for Perceptron:

1- Run `main.py` and pass the training, vocab & labels files as follows:

```
python main.py --class 0 --train [training_file] --vocab [vocab_file] --labels [labels_file]
```
This will produce several files in the `model` directory (if they are not already there), those files will be used to run the tagger on the testing file



3- Run `main.py` by passing the testing file as follows:

```
python main.py --class 0 --test [testing_file]
```
This will produce the final tagged file in the `data` directory

# Steps for HMM:

1- Run `main.py` and pass the training, vocab & labels files as follows:

```
python main.py --class 1 --train [training_file] --freq [freq_file]
```

2- Run `main.py` by passing the testing file as follows:

```
python main.py --class 1 --test [testing_file] --freq [freq_file]
```

Where freq is the frequency file produced by ./prepare.sh
###### Examples on how to run the tool is availble in the scripts directory ######


# Usage:

Options for running the tagger

```
usage: main.py [-h] [--class CLASSIFIER] [--train TRAIN] [--step STEP]
               [--iter ITER] [--vocab VOCAB] [--labels LABELS] [--freq FREQ]
               [--test TEST] [--eval {0,1}] [--pred PRED] [--gold GOLD]

optional arguments:
  -h, --help          show this help message and exit
  --class CLASSIFIER  0: Perceptron, 1: HMM
  --train TRAIN       Training file
  --step STEP         Decaying Step Size
  --iter ITER         Training iterations
  --vocab VOCAB       Vocab file - For Perceptron Training & Evaluation
  --labels LABELS     Labels file - For Perceptron Training & Evaluation
  --freq FREQ         Frequency file - For HMM Training
  --test TEST         Testing file
  --eval {0,1}        Set to 1, to run evaluation [default=0]. The following
                      files must be specified: (Vocab, Labels, Pred, Gold)
  --pred PRED         In case eval was set to 1: Tagged Predictions File
  --gold GOLD         In case eval was set to 1: Gold Standard File
```

