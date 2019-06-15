#!/bin/sh

mkdir data
mkdir data/cache
mkdir data/models

cd data

lfw_url=http://vis-www.cs.umass.edu/lfw/lfw.tgz
lfw_funnel_url=http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
pairs_url=http://vis-www.cs.umass.edu/lfw/pairs.txt

wget ${lfw_url}
wget ${lfw_funnel_url}
wget ${pairs_url}

tar -xzvf lfw.tar.gz
tar -xzvf lfw-deepfunneled.tgz

rm lfw.tar.gz
rm lfw-deepfunneled.tgz
