#!/bin/bash -l
#EXPECTS TO BE RUN FROM OUTSIDE OF CAFFE REPO (AKA ../..)
cp -rp caffe caffe_built
cd caffe_built
export PATH=/usr/common/software/python/2.7-anaconda/envs/caffe_env/bin/:$PATH
make clean
make all -j 12
make pycaffe
cd ..
