python -B main.py --extract 1 --i data/dev.col --o models/train-feats.txt
python -B main.py --train-perceptron 1 --i data/dev.col --feats models/train-feats.txt --labels data/train.col.labels --step 1 --iter 10
python -B main.py --extract 1 --i data/dev.col --o models/dev-feats.txt
python -B main.py --test-perceptron 1 --i data/dev.col --feats models/dev-feats.txt --labels data/train.col.labels --o models/pred.col
python -B main.py --eval 1 --gold data/dev.col --pred models/pred.col --labels data/train.col.labels --o models/eval.out
