file=$(basename $1)
path=$(dirname $1)

vocab=$file.vocab
labels=$file.labels

cat $1 | cut -d$'\t' -f1 | sort --uniq | sed '1d' &> $path/$vocab
cat $1 | cut -d$'\t' -f2 | sort --uniq | sed '1d'&> $path/$labels
