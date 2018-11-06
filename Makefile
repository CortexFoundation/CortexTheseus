# This Makefile is meant to be used by people that do not usually work
# with Go source code. If you know what GOPATH is then you probably
# don't need to bother with make.

GOBIN = $(shell pwd)/build/bin
GO ?= latest
LIB_CUCKOO_DIR = $(shell pwd)/miner/libcuckoo

all: cuda-miner opencl-miner

cuda-miner: clib
	go build -o build/bin/cuda_miner ./cmd/miner 
	@echo "Done building."

opencl-miner: clib
	go build -o build/bin/opencl_miner -tags opencl ./cmd/miner 
	@echo "Done building."

clib:
	make -C $(LIB_CUCKOO_DIR)

clean:
	rm -fr build/_workspace/pkg/ $(GOBIN)/*
	go clean -cache

