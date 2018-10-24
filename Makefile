# This Makefile is meant to be used by people that do not usually work
# with Go source code. If you know what GOPATH is then you probably
# don't need to bother with make.

GOBIN = $(shell pwd)/build/bin
GO ?= latest
LIB_MINER_DIR = $(shell pwd)/cminer/
LIB_CUDA_MINER_DIR = $(shell pwd)/miner/cuckoocuda

cuckoo-miner: clib
	go build -o build/bin/miner ./cmd/miner 
	@echo "Done building."

clib:
	make -C $(LIB_CUDA_MINER_DIR)

clean:
	rm -fr build/_workspace/pkg/ $(GOBIN)/*

