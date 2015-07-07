python main.py --class 0 --train data/train.col --vocab data/train.col.vocab --labels data/train.col.labels --step 0.1 --iter 10
python main.py --class 0 --test data/dev.col
python main.py --eval 1 --vocab data/train.col.vocab --labels data/train.col.labels --pred data/pred.col --gold data/dev.col
