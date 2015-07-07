file=$(basename $1)
path=$(dirname $1)
home=$(pwd)

vocab=$file.vocab
labels=$file.labels
freq=$file.freq

cat $1 | cut -d$'\t' -f1 | sort | uniq -c | sort -nr &> $path/$freq
cat $1 | cut -d$'\t' -f1 | sort --uniq | sed '1d' | tr '[:upper:]' '[:lower:]' &> $path/$vocab
cat $1 | cut -d$'\t' -f2 | sort --uniq | sed '1d' &> $path/$labels




