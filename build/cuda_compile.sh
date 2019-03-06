#!/bin/bash

HOST_ID=$(lsb_release -i | awk '{print $3}')
HOST_VERSION=$(lsb_release -r | awk '{print $2}')
# CUDA_VERSION=10.0

sudo unlink /usr/local/cuda
sudo ln -s /usr/local/cuda-${CUDA_VERSION} /usr/local/cuda
source ~/.bashrc

# print nvcc version
echo "CUDA VERSION"
nvcc --version
echo
echo

make clean-all
make cortex -j4
mv build/bin/cortex cortex-${HOST_ID}-${HOST_VERSION}-cuda-${CUDA_VERSION}
