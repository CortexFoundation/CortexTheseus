# This Makefile is meant to be used by people that do not usually work
# with Go source code. If you know what GOPATH is then you probably
# don't need to bother with make.

.PHONY: cortex android ios cortex-cross cvm all test clean
.PHONY: cortex-linux cortex-linux-386 cortex-linux-amd64 cortex-linux-mips64 cortex-linux-mips64le
.PHONY: cortex-linux-arm cortex-linux-arm-5 cortex-linux-arm-6 cortex-linux-arm-7 cortex-linux-arm64
.PHONY: cortex-darwin cortex-darwin-386 cortex-darwin-amd64
.PHONY: cortex-windows cortex-windows-386 cortex-windows-amd64

.PHONY: clib cvm_runtime
.PHONY: cortex

BASE = $(shell pwd)
GOBIN = $(shell pwd)/build/bin
GO ?= latest
LIB_MINER_DIR = $(shell pwd)/solution
INFER_NET_DIR = $(shell pwd)/cvm-runtime

# Curkoo algorithm dynamic library path
OS = $(shell uname)
ifeq ($(OS), Linux)
endif

ifeq ($(OS), Darwin)
endif
cortex: cpu

all: cortex bootnode abigen devp2p keytools rlpdump wnode

gpu: cortex_gpu

cpu: cortex_cpu

mine: cortex_mine

#format:
#	find . -name '*.go' -type f -not -path "./vendor*" -not -path "*.git*" -not -path "*/generated/*" | xargs gofmt -w -s

submodule:
	build/env.sh

clean-miner: submodule
	rm -fr plugins/*_helper_for_node.so

cortex_cpu: clean-miner clib_cpu
	build/env.sh go run build/ci.go install ./cmd/cortex
	@echo "Done building."
	@echo "Run \"$(GOBIN)/cortex\" to launch cortex cpu."
cortex_mine: clean-miner clib_mine
	build/env.sh go run build/ci.go install ./cmd/cortex
	@echo "Done building."
	@echo "Run \"$(GOBIN)/cortex\" to launch cortex miner."
cortex_gpu: clean-miner clib
	build/env.sh go run build/ci.go install ./cmd/cortex
	@echo "Done building."
	@echo "Run \"$(GOBIN)/cortex\" to launch cortex gpu."
bootnode:
	build/env.sh go run build/ci.go install ./cmd/bootnode
	@echo "Done building."
	@echo "Run \"$(GOBIN)/bootnode\" to launch cortex bootnode."
abigen:
	build/env.sh go run build/ci.go install ./cmd/abigen
	@echo "Done building."
	@echo "Run \"$(GOBIN)/abigen\" to launch abigen."
devp2p:
	build/env.sh go run build/ci.go install ./cmd/devp2p
	@echo "Done building."
	@echo "Run \"$(GOBIN)/devp2p\" to launch cortex devp2p."
keytools:
	build/env.sh go run build/ci.go install ./cmd/keytools
	@echo "Done building."
	@echo "Run \"$(GOBIN)/keytools\" to launch cortex keytools."
rlpdump:
	build/env.sh go run build/ci.go install ./cmd/rlpdump
	@echo "Done building."
	@echo "Run \"$(GOBIN)/rlpdump\" to launch cortex rlpdump."
wnode:
	build/env.sh go run build/ci.go install ./cmd/wnode
	@echo "Done building."
	@echo "Run \"$(GOBIN)/wnode\" to launch cortex whisper node."

torrent-test:
	build/env.sh go run build/ci.go install ./cmd/torrent-test
	@echo "Done building."
	@echo "Run \"$(GOBIN)/torrent-test\" to launch cortex torrentfs-test."

cvm: plugins/libcvm_runtime.so
	build/env.sh go run build/ci.go install ./cmd/cvm
	@echo "Done building."
	@echo "Run \"$(GOBIN)/cvm\" to launch cortex vm."
nodekey:
	build/env.sh go run build/ci.go install ./cmd/nodekey
	@echo "Done building."
	@echo "Run \"$(GOBIN)/nodekey\" to launch nodekey."

plugins/cuda_helper_for_node.so: 
	$(MAKE) -C $(BASE)/solution cuda
	build/env.sh go build -buildmode=plugin -o $@ consensus/cuckoo/plugins/cuda/cuda_helper_for_node.go

plugins/cpu_helper_for_node.so:
	$(MAKE) -C $(BASE)/solution cpu

plugins/libcvm_runtime.so: submodule
	$(MAKE) -C ${INFER_NET_DIR} -j8 lib
	@mkdir -p plugins
	ln -sf ${INFER_NET_DIR}/build/libcvm_runtime.so $@

clib_cpu: plugins/cpu_helper_for_node.so plugins/libcvm_runtime.so

clib: plugins/cuda_helper_for_node.so plugins/cpu_helper_for_node.so plugins/libcvm_runtime.so

clib_mine: plugins/cuda_helper_for_node.so plugins/cpu_helper_for_node.so plugins/libcvm_runtime.so

#inferServer: clib
#	build/env.sh go run build/ci.go install ./cmd/infer_server
#	build/env.sh go run build/ci.go install ./cmd/infer_client

android:
	build/env.sh go run build/ci.go aar --local
	@echo "Done building."
	@echo "Import \"$(GOBIN)/cortex.aar\" to use the library."
	@echo "Import \"$(GOBIN)/cortex-sources.jar\" to add javadocs"
	@echo "For more info see https://stackoverflow.com/questions/20994336/android-studio-how-to-attach-javadoc"

ios:
	build/env.sh go run build/ci.go xcode --local
	@echo "Done building."
	@echo "Import \"$(GOBIN)/Ctxc.framework\" to use the library."

lint: ## Run linters.
	build/env.sh go run build/ci.go lint

clean: clean-clib
	./build/clean_go_build_cache.sh
	rm -fr build/_workspace/pkg/ $(GOBIN)/* plugins/* build/_workspace/src/ #solution/*.a solution/*.o
	# rm -rf inference/synapse/kernel
	# ln -sf ../../cvm-runtime/kernel inference/synapse/kernel

clean-clib:
	$(MAKE) -C $(LIB_MINER_DIR) clean
	$(MAKE) -C $(INFER_NET_DIR) clean
	
.PHONY: clean-all
clean-all: clean-clib clean

# The devtools target installs tools required for 'go generate'.
# You need to put $GOBIN (or $GOPATH/bin) in your PATH to use 'go generate'.

devtools:
	env GOBIN= go get -u golang.org/x/tools/cmd/stringer@latest
	env GOBIN= go get -u github.com/fjl/gencodec@latest
	env GOGIN= go get -u google.golang.org/protobuf/cmd/protoc-gen-go@latest
	env GOBIN= go install ./cmd/abigen
	@type "npm" 2> /dev/null || echo 'Please install node.js and npm'
	@type "solc" 2> /dev/null || echo 'Please install solc'
	@type "protoc" 2> /dev/null || echo 'Please install protoc'

# Cross Compilation Targets (xgo)

cortex-cross: cortex-linux cortex-darwin cortex-windows cortex-android cortex-ios
	@echo "Full cross compilation done:"
	@ls -ld $(GOBIN)/cortex-*

cortex-linux: cortex-linux-386 cortex-linux-amd64 cortex-linux-arm cortex-linux-mips64 cortex-linux-mips64le
	@echo "Linux cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-*

cortex-linux-386:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/386 -v ./cmd/cortex
	@echo "Linux 386 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep 386

cortex-linux-amd64:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/amd64 -v ./cmd/cortex
	@echo "Linux amd64 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep amd64

cortex-linux-arm: cortex-linux-arm-5 cortex-linux-arm-6 cortex-linux-arm-7 cortex-linux-arm64
	@echo "Linux ARM cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep arm

cortex-linux-arm-5:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/arm-5 -v ./cmd/cortex
	@echo "Linux ARMv5 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep arm-5

cortex-linux-arm-6:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/arm-6 -v ./cmd/cortex
	@echo "Linux ARMv6 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep arm-6

cortex-linux-arm-7:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/arm-7 -v ./cmd/cortex
	@echo "Linux ARMv7 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep arm-7

cortex-linux-arm64:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/arm64 -v ./cmd/cortex
	@echo "Linux ARM64 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep arm64

cortex-linux-mips:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/mips --ldflags '-extldflags "-static"' -v ./cmd/cortex
	@echo "Linux MIPS cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep mips

cortex-linux-mipsle:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/mipsle --ldflags '-extldflags "-static"' -v ./cmd/cortex
	@echo "Linux MIPSle cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep mipsle

cortex-linux-mips64:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/mips64 --ldflags '-extldflags "-static"' -v ./cmd/cortex
	@echo "Linux MIPS64 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep mips64

cortex-linux-mips64le:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=linux/mips64le --ldflags '-extldflags "-static"' -v ./cmd/cortex
	@echo "Linux MIPS64le cross compilation done:"
	@ls -ld $(GOBIN)/cortex-linux-* | grep mips64le

cortex-darwin: cortex-darwin-386 cortex-darwin-amd64
	@echo "Darwin cross compilation done:"
	@ls -ld $(GOBIN)/cortex-darwin-*

cortex-darwin-386:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=darwin/386 -v ./cmd/cortex
	@echo "Darwin 386 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-darwin-* | grep 386

cortex-darwin-amd64:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=darwin/amd64 -v ./cmd/cortex
	@echo "Darwin amd64 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-darwin-* | grep amd64

cortex-windows: cortex-windows-386 cortex-windows-amd64
	@echo "Windows cross compilation done:"
	@ls -ld $(GOBIN)/cortex-windows-*

cortex-windows-386:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=windows/386 -v ./cmd/cortex
	@echo "Windows 386 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-windows-* | grep 386

cortex-windows-amd64:
	build/env.sh go run build/ci.go xgo -- --go=$(GO) --targets=windows/amd64 -v ./cmd/cortex
	@echo "Windows amd64 cross compilation done:"
	@ls -ld $(GOBIN)/cortex-windows-* | grep amd64
