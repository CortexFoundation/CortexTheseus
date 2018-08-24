#!/bin/bash

# Curkoo algorithm dynamic library path
OS=`uname`
ROOT_DIR=`pwd`
LIB_MINER_DIR=$ROOT_DIR/libminer

if [[ "$OS" == "Linux" ]]; then
    export LD_LIBRARY_PATH=$LIB_MINER_DIR/linux/
    export CGO_CFLAGS=-I$LIB_MINER_DIR/
    export CGO_LDFLAGS=-L$LIB_MINER_DIR/linux/
    echo $LD_LIBRARY_PATH
elif [[ "$OS" == "Darwin" ]]; then
	export DYLD_LIBRARY_PATH=$LIB_MIER_DIR/mac/
    export CGO_CFLAGS=-I$LIB_MINER_DIR/
    export CGO_LDFLAGS=-L$LIB_MINER_DIR/mac/
	echo $DYLD_LIBRARY_PATH
else
	echo "Sorry, cannot detected your OS" $OS
fi

