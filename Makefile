# This Makefile is meant to be used by people that do not usually work
# with Go source code. If you know what GOPATH is then you probably
# don't need to bother with make.

GOBIN = $(shell pwd)/build/bin
GO ?= latest
LIB_CUCKOO_DIR = $(shell pwd)/miner/libcuckoo

all: cuda-miner opencl-miner cuckaroo-miner

cuda-miner: 
	make -C ${LIB_CUCKOO_DIR} cuda
	go build -o build/bin/cuda_miner -tags cuda  ./cmd/miner 
	@echo "Done building."

opencl-miner: 
	make -C ${LIB_CUCKOO_DIR} opencl
	go build -o build/bin/opencl_miner -tags opencl ./cmd/miner 
	@echo "Done building."

cuckaroo-miner: 
	make -C ${LIB_CUCKOO_DIR} cuckaroo
	go build -o build/bin/cuckaroo_miner -tags cuckaroo ./cmd/miner 
	@echo "Done building."

clean:
	rm -fr build/_workspace/pkg/ $(GOBIN)/*
	go clean -cache

