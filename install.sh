#!/bin/sh

if [ "$1" == "" ]; then
	echo "Usage $0 executable branch ethereum develop"
	echo "executable    ethereum or mist"
	echo "branch        develop or master"
	exit
fi

exe=$1
branch=$2

# Test if go is installed
command -v go >/dev/null 2>&1 || { echo >&2 "Unable to find 'go'. This script requires go."; exit 1; }

# Test if $GOPATH is set
if [ "$GOPATH" == "" ]; then
	echo "\$GOPATH not set"
	exit
fi

echo "go get -u -d github.com/ethereum/go-ethereum/$exe"
go get -v -u -d github.com/ethereum/go-ethereum/$exe
if [ $? != 0 ]; then
	echo "go get failed"
	exit
fi

echo "eth-go"
cd $GOPATH/src/github.com/ethereum/go-ethereum
git checkout $branch

echo "go-ethereum"
cd $GOPATH/src/github.com/ethereum/go-ethereum/$exe
git checkout $branch

if [ "$exe" == "mist" ]; then
	echo "Building Mist GUI. Assuming Qt is installed. If this step"
	echo "fails; please refer to: https://github.com/ethereum/go-ethereum/wiki/Building-Ethereum(Go)"
else
	echo "Building ethereum CLI."
fi

go install
if [ $? == 0 ]; then
	echo "go install failed"
	exit
fi

echo "done. Please run $exe :-)"
