python main.py --class 1 --train data/train.col --freq data/train.col.freq
python main.py --class 1 --test data/dev.col --freq data/train.col.freq
python main.py --eval 1 --vocab data/train.col.vocab --labels data/train.col.labels --pred data/pred.col --gold data/dev.col
