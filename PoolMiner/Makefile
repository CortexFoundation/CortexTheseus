# This Makefile is meant to be used by people that do not usually work
# with Go source code. If you know what GOPATH is then you probably
# don't need to bother with make.

GOBIN = $(shell pwd)/build/bin
GO ?= latest
LIB_CUCKOO_DIR = $(shell pwd)/miner/libcuckoo
PLUGINS_DIR = $(shell pwd)/plugins

all:
	make -C ${LIB_CUCKOO_DIR}
	go build -buildmode=plugin -o ${PLUGINS_DIR}/cuda_helper.so ./miner/libcuckoo/cuda_helper.go
	go build -buildmode=plugin -o ${PLUGINS_DIR}/opencl_helper.so ./miner/libcuckoo/opencl_helper.go
	go build -buildmode=plugin -o ${PLUGINS_DIR}/cpu_helper.so ./miner/libcuckoo/cpu_helper.go
	go build -o build/bin/cortex_miner ./cmd/miner

cuda-miner: 
	make -C ${LIB_CUCKOO_DIR} cuda
	go build -buildmode=plugin -o ${PLUGINS_DIR}/cuda_helper.so ./miner/libcuckoo/cuda_helper.go
	go build -o build/bin/cortex_miner  ./cmd/miner
	@echo "Done building."

opencl-miner: 
	make -C ${LIB_CUCKOO_DIR} opencl
	go build -buildmode=plugin -o ${PLUGINS_DIR}/opencl_helper.so ./miner/libcuckoo/opencl_helper.go
	go build -o build/bin/cortex_miner ./cmd/miner
	@echo "Done building."

cpu-miner: 
	make -C ${LIB_CUCKOO_DIR} cpu
	go build -buildmode=plugin -o ${PLUGINS_DIR}/cpu_helper.so ./miner/libcuckoo/cpu_helper.go
	go build -o build/bin/cortex_miner ./cmd/miner
	@echo "Done building."

clean:
	rm -fr build/_workspace/pkg/ $(GOBIN)/* $(PLUGINS_DIR)/*
	go clean -cache

