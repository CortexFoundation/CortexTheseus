#!/bin/bash

OS=`uname`
ROOT_DIR=`pwd`

# Curkoo algorithm dynamic library path
# LIB_MINER_DIR=$ROOT_DIR/cminer
INFER_NET_DIR=$ROOT_DIR/infernet

if [[ "$OS" -eq "Linux" ]]; then
    # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_MINER_DIR/:$LIB_MINER_DIR/cuckoo:$INFER_NET_DIR
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_MINER_DIR/cuckoo:$INFER_NET_DIR
    # export CGO_CFLAGS=-I$LIB_MINER_DIR/
    # export CGO_LDFLAGS=-L$LIB_MINER_DIR/\ -L$LIB_MINER_DIR/cuckoo
    echo $LD_LIBRARY_PATH
    # echo $CGO_LDFLAGS
elif [[ "$OS" -eq "Darwin" ]]; then
	#export DYLD_LIBRARY_PATH=$LIB_MINER_DIR/:$LIB_MINER_DIR/cuckoo
    # export CGO_CFLAGS=-I$LIB_MINER_DIR/
    # export CGO_LDFLAGS=-L$LIB_MINER_DIR/\ -L$LIB_MINER_DIR/cuckoo
	echo $DYLD_LIBRARY_PATH
else
	echo "Sorry, cannot detected your OS" $OS
fi

