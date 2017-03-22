#!/bin/bash -l
#EXPECTS TO BE RUN FROM OUTSIDE OF CAFFE REPO (AKA ../..)
module swap PrgEnv-cray PrgEnv-intel
rm -rf caffe_built
mkdir caffe_built
cd caffe
cp -r `ls -A  | grep -v ".git"` ../caffe_built
cd ../caffe_built
export PATH=/usr/common/software/python/2.7-anaconda/envs/caffe/bin/:$PATH
make clean
make all -j 12
make pycaffe CFLAGS='-w'

