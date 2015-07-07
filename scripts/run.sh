python main.py --class 0 --train data/dev.col --vocab data/dev.col.vocab --labels data/dev.col.labels --step 0.1 --iter 10
python main.py --class 0 --test data/dev.col
python main.py --eval 1 --vocab data/dev.col.vocab --labels data/dev.col.labels --pred data/pred.col --gold data/dev.col
