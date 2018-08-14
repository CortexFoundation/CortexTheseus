#!/bin/bash

# Curkoo algorithm dynamic library path
OS=`uname`
ROOT_DIR=`pwd`

if [[ "$OS" == "Linux" ]]; then
	export LD_LIBRARY_PATH=$ROOT_DIR
elif [[ "$OS" == "Darwin" ]]; then
	export DYLD_LIBRARY_PATH=$ROOT_DIR
	echo $DYLD_LIBRARY_PATH
else
	echo "Sorry, cannot detected your OS" $OS
fi

export CGO_CFLAGS=-I$ROOT_DIR/
export CGO_LDFLAGS=-L$ROOT_DIR/
