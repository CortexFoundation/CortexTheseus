#!/bin/bash

# Curkoo algorithm dynamic library path
OS=`uname`
ROOT_DIR=`pwd`
export LIB_MINER_DIR=$ROOT_DIR/cminer

if [[ "$OS" == "Linux" ]]; then
    export LD_LIBRARY_PATH=$LIB_MINER_DIR/
    export CGO_CFLAGS=-I$LIB_MINER_DIR/
    export CGO_LDFLAGS=-L$LIB_MINER_DIR/
    echo $LD_LIBRARY_PATH
elif [[ "$OS" == "Darwin" ]]; then
	export DYLD_LIBRARY_PATH=$LIB_MINER_DIR/
    export CGO_CFLAGS=-I$LIB_MINER_DIR/
    export CGO_LDFLAGS=-L$LIB_MINER_DIR/\ -L$LIB_MINER_DIR/cuckoo
	echo $DYLD_LIBRARY_PATH
else
	echo "Sorry, cannot detected your OS" $OS
fi

