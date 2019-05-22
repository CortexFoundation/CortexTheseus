.POSIX:
.SUFFIXES:

OPT ?= -O3

GCC_ARCH_FLAGS ?= -march=native
GPP_ARCH_FLAGS ?= -march=native

# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS ?= -O3 -Wall -Wno-format -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L $(OPT) -DPREFETCH -I. -Isrc/opencl/ocl/include/ $(CPPFLAGS) -pthread -fPIC
GPP ?= g++ $(GPP_ARCH_FLAGS) -std=c++11 $(FLAGS)
CFLAGS ?= -Wall -Wno-format -fomit-frame-pointer $(OPT)
GCC ?= gcc $(GCC_ARCH_FLAGS) -std=gnu11 $(CFLAGS)

PROOFSIZE=42
EDGEBITS=28

all: cuda opencl cpu

cuda : libcudaminer.a
	
opencl : libopenclminer.a

cpu : libcpuminer.a

blake2b.o: src/blake2.h src/blake2-impl.h src/blake2b-ref.cpp
	$(GPP) -c -o blake2b.o src/blake2b-ref.cpp

cuckoo.o: src/cuckoo.cc src/cuckoo.h
	$(GPP) -c -o $@ -DPROOFSIZE=$(PROOFSIZE) -DEDGEBITS=$(EDGEBITS) $<

siphash.o: src/siphash.cc src/siphash.h
	$(GPP) -c -o $@ $<


cudaN31L14.o.trimmer.o: src/siphash.cuh src/cuda/cuckoo_solver.hpp src/cuda/cuckaroo_solver.hpp src/cuda/miner.cu src/cuda/trimmer.cu src/cuda/monitor.hpp Makefile
	nvcc -Xcompiler="-fPIC" -std=c++11 -O3 -c -o $@ -rdc=true -DEDGEBITS=${EDGEBITS} -DPROOFSIZE=${PROOFSIZE} -arch sm_35 -D_FORCE_INLINES src/cuda/trimmer.cu $(LIBS)

cudaN31L14.o.mean.o: src/siphash.cuh src/cuda/cuckoo_solver.hpp src/cuda/cuckaroo_solver.hpp src/cuda/miner.cu src/cuda/trimmer.cu src/cuda/monitor.hpp Makefile
	nvcc -Xcompiler="-fPIC" -std=c++11 -O3 -c -o $@ -rdc=true -DEDGEBITS=${EDGEBITS} -DPROOFSIZE=${PROOFSIZE} -arch sm_35 -D_FORCE_INLINES src/cuda/miner.cu $(LIBS) -lnvidia-ml

cudaN31L14.o: cudaN31L14.o.trimmer.o cudaN31L14.o.mean.o
	@echo 'NVCC '$@
	nvcc -Xcompiler="-fPIC" -ccbin g++ -O3 -std=c++11 -m64 -Xcompiler \
		-fpermissive -gencode arch=compute_35,code=sm_35 -D_FORCE_INLINES \
		-rdc=true -dlink -o $@ $^ 

openclN31L14.o.mean.o: src/siphash.cuh src/opencl/cuckoo_solver.hpp src/opencl/cuckaroo_solver.hpp src/opencl/miner.cpp src/opencl/trimmer.cpp src/opencl/monitor.hpp Makefile
	$(GPP) -c -U__STRICT_ANSI__ -o $@ -DEDGEBITS=${EDGEBITS} -DPROOFSIZE=${PROOFSIZE} src/opencl/trimmer.cpp

openclN31L14.o.temp.o: src/siphash.cuh src/opencl/cuckoo_solver.hpp src/opencl/cuckaroo_solver.hpp src/opencl/miner.cpp src/opencl/trimmer.cpp src/opencl/monitor.hpp Makefile
	$(GPP) -c -U__STRICT_ANSI__ -o $@ -DEDGEBITS=${EDGEBITS} -DPROOFSIZE=${PROOFSIZE} src/opencl/miner.cpp -lrocm_smi64 -lOpenCL

openclN31L14.o: openclN31L14.o.mean.o openclN31L14.o.temp.o
	#$(GPP) -c -U__STRICT_ANSI__ -o $@.mean.o -DEDGEBITS=${EDGEBITS} -DPROOFSIZE=${PROOFSIZE} src/opencl/trimmer.cpp
#has rocm_smi_lib
#	$(GPP) -g -c -U__STRICT_ANSI__ -DPLATFORM_AMD -o $@.temp.o -DEDGEBITS=${EDGEBITS} -DPROOFSIZE=${PROOFSIZE} src/opencl/miner.cpp -lrocm_smi64
#no rocm_smi_lib
	ld -r -o $@ $^ 

cpuminer.o:  src/cpu/bitmap.hpp src/cpu/graph.hpp src/cpu/barrier.hpp src/cpu/siphash.hpp src/cpu/cuckaroo_mean.hpp src/cpu/cuckoo_mean.hpp src/cpu/miner.cpp  Makefile
	$(GPP) -c -o $@.temp.o -mavx2 -DNSIPHASH=8 -DEDGEBITS=${EDGEBITS} -DPROOFSIZE=${PROOFSIZE} src/cpu/miner.cpp 
	ld -r -o $@ $@.temp.o

libcudaminer.a: cudaN31L14.o cudaN31L14.o.trimmer.o cudaN31L14.o.mean.o blake2b.o siphash.o cuckoo.o
	ar cr $@ $^ # $<.trimmer.o $<.miner.o 
	# rm -rf cudaN31L14*.o
#	$(GPP) -g -o $@ $^ $<.temp.o  $<.mean.o -lOpenCL
#	ar cr $@ $^

libopenclminer.a: openclN31L14.o blake2b.o siphash.o cuckoo.o
	ar cr $@ $^ $<.temp.o  $<.mean.o
	# rm -rf openclN31L14*.o

libcpuminer.a: cpuminer.o blake2b.o siphash.o cuckoo.o
	ar cr $@ $^
	# rm -rf cpuminer*.o

clean:
	rm -rf *.o *.a
