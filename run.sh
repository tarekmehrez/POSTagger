python main.py --train data/dev.col --vocab data/dev.col.vocab --labels data/dev.col.labels
python main.py --test data/dev.col
python main.py --eval 1 --vocab data/dev.col.vocab --labels data/dev.col.labels --pred data/pred.col --gold data/dev.col
