# This Makefile is meant to be used by people that do not usually work
# with Go source code. If you know what GOPATH is then you probably
# don't need to bother with make.

CBIN = $(shell pwd)/build/bin
GO ?= latest
LIB_CUCKOO_DIR = $(shell pwd)/miner/libcuckoo
PLUGINS_DIR = $(shell pwd)/plugins
#CORTEXPATH = $(shell pwd)/../build/_workspace

all:
	build/env.sh go get -tags remote -v ./...
	$(MAKE) -C ${LIB_CUCKOO_DIR}
	build/env.sh go build -buildmode=plugin -o ${PLUGINS_DIR}/cuda_helper.so ./miner/libcuckoo/cuda_helper.go
	#build/env.sh go build -buildmode=plugin -o ${PLUGINS_DIR}/opencl_helper.so ./miner/libcuckoo/opencl_helper.go
	build/env.sh go build -buildmode=plugin -o ${PLUGINS_DIR}/cpu_helper.so ./miner/libcuckoo/cpu_helper.go
	build/env.sh go build -o build/bin/cortex_miner ./cmd/miner

cuda-miner:
	build/env.sh go get -tags remote -v ./...
	$(MAKE) -C ${LIB_CUCKOO_DIR} cuda
	build/env.sh  go build -buildmode=plugin -o ${PLUGINS_DIR}/cuda_helper.so ./miner/libcuckoo/cuda_helper.go
	build/env.sh go build -o build/bin/cortex_miner  ./cmd/miner
	@echo "Done building."

opencl-miner: 
	build/env.sh go get -tags remote -v ./...
	$(MAKE) -C ${LIB_CUCKOO_DIR} opencl
	build/env.sh go build -buildmode=plugin -o ${PLUGINS_DIR}/opencl_helper.so ./miner/libcuckoo/opencl_helper.go
	build/env.sh go build -o build/bin/cortex_miner ./cmd/miner
	@echo "Done building."

cpu-miner: 
	build/env.sh go get -tags remote -v ./...
	$(MAKE) -C ${LIB_CUCKOO_DIR} cpu
	build/env.sh go build -buildmode=plugin -o ${PLUGINS_DIR}/cpu_helper.so ./miner/libcuckoo/cpu_helper.go
	build/env.sh go build -o build/bin/cortex_miner ./cmd/miner
	@echo "Done building."

clean:
	rm miner/libcuckoo/*.o miner/libcuckoo/*.a
	rm -fr build/_workspace/pkg/ $(CBIN)/* $(PLUGINS_DIR)/*
	build/env.sh go clean -cache

