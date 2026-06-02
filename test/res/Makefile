# Fetches test datasets into this directory (test/res).
# Data is not committed (size + licensing); fetch on demand.
#
# Usage (run from test/res):
#   make <dataset>     e.g. make conll2003
#   make all
#
# Each target downloads upstream data and transforms it into the exact
# layout/format the loaders in lib/santoku/learn/dataset.lua expect.

CURL := curl -fL
LUA  := lua

URL_CONLL2003          := https://data.deepai.org/conll2003.zip
URL_20NEWS             := http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz
URL_MNIST_TRAIN        := https://pjreddie.com/media/files/mnist_train.csv
URL_MNIST_TEST         := https://pjreddie.com/media/files/mnist_test.csv
URL_IMDB               := https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
URL_CALIFORNIA_HOUSING := https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz
URL_EURLEX57K          := https://archive.org/download/EURLEX57K

.PHONY: all conll2003 20news mnist imdb eurlex57k california-housing

all: conll2003 20news mnist imdb eurlex57k california-housing

conll2003:
	$(CURL) $(URL_CONLL2003) -o conll2003.zip
	rm -rf conll2003 && unzip -q conll2003.zip -d conll2003
	rm -f conll2003.zip

20news:
	$(CURL) $(URL_20NEWS) -o 20news-bydate.tar.gz
	tar -xzf 20news-bydate.tar.gz

# 784 binarized pixels (value/255 > 0.3 -> 1, else 0) then the label,
# space-separated; train then test = 70k rows.
mnist:
	$(CURL) $(URL_MNIST_TRAIN) -o mnist_train.csv
	$(CURL) $(URL_MNIST_TEST) -o mnist_test.csv
	awk -F, '{ out=""; for (i=2;i<=NF;i++) out=out ($$i/255>0.3?1:0)" "; print out $$1 }' \
	  mnist_train.csv mnist_test.csv > mnist.70k.txt
	rm -f mnist_train.csv mnist_test.csv

# Merge train+test per sentiment into imdb.50k/{pos,neg}; prefix the original
# filenames with the split so nothing collides (full 50k = 12500 x 4).
# Single Lua process; renames into place (aclImdb is removed after).
imdb:
	$(CURL) $(URL_IMDB) -o aclImdb_v1.tar.gz
	tar -xzf aclImdb_v1.tar.gz
	rm -rf imdb.50k
	$(LUA) imdb_merge.lua aclImdb imdb.50k
	rm -rf aclImdb aclImdb_v1.tar.gz

# Splits are published directly as jsonl on the archive item.
eurlex57k:
	rm -rf eurlex57k && mkdir -p eurlex57k
	$(CURL) $(URL_EURLEX57K)/train.jsonl -o eurlex57k/train.jsonl
	$(CURL) $(URL_EURLEX57K)/dev.jsonl   -o eurlex57k/dev.jsonl
	$(CURL) $(URL_EURLEX57K)/test.jsonl  -o eurlex57k/test.jsonl

# Stock handson-ml housing.csv, renamed.
california-housing:
	$(CURL) $(URL_CALIFORNIA_HOUSING) -o housing.tgz
	tar -xzf housing.tgz
	mv -f housing.csv california-housing.csv
	rm -f housing.tgz
