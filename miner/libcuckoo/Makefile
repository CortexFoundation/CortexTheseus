.POSIX:
.SUFFIXES:

OPT ?= -O3

GCC_ARCH_FLAGS ?= -march=native
GPP_ARCH_FLAGS ?= -march=native

# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS ?= -O0 -Wall -Wno-format -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L $(OPT) -DPREFETCH -I. -Isrc/opencl/ocl/include/ $(CPPFLAGS) -pthread -lOpenCL -g
GPP ?= g++ $(GPP_ARCH_FLAGS) -std=c++11 $(FLAGS)
CFLAGS ?= -Wall -Wno-format -fomit-frame-pointer $(OPT)
GCC ?= gcc $(GCC_ARCH_FLAGS) -std=gnu11 $(CFLAGS)

all: cuda opencl

cuda : libcudaminer.a
	
opencl : libopenclminer.a

blake2b.o: src/blake2.h src/blake2-impl.h src/blake2b-ref.cpp
	$(GPP) -g -Og -c -o blake2b.o src/blake2b-ref.cpp

cuckoo.o: src/cuckoo.cc src/cuckoo.h
	$(GPP) -g -c -o $@ $<

siphash.o: src/siphash.cc src/siphash.h
	$(GPP) -g -c -o $@ $<


cudaN31L14.o: src/siphash.cuh src/cuda/cuckoo_solver.hpp src/cuda/cuckaroo_solver.hpp src/cuda/miner.cu src/cuda/trimmer.cu src/cuda/monitor.hpp Makefile
	nvcc -g -std=c++11 -c -o $@.trimmer.o -rdc=true -DEDGEBITS=29 -DPROOFSIZE=42 -arch sm_35 src/cuda/trimmer.cu $(LIBS)
#	nvcc -g -c -o $@.solver.o -rdc=true -DEDGEBITS=28 -DPROOFSIZE=12 -arch sm_35 src/cuda/solver.hpp $(LIBS)
	nvcc -g -std=c++11 -c -o $@.miner.o -rdc=true -DEDGEBITS=29 -DPROOFSIZE=42 -arch sm_35 src/cuda/miner.cu $(LIBS) -lnvidia-ml
	nvcc -g -ccbin g++ -std=c++11 -m64 -Xcompiler -fpermissive -gencode arch=compute_35,code=sm_35 -rdc=true -dlink -o $@ $@.miner.o $@.trimmer.o

openclN31L14.o: src/siphash.cuh src/cuda/cuckoo_solver.hpp src/cuda/cuckaroo_solver.hpp src/opencl/miner.cpp src/opencl/trimmer.cpp src/opencl/monitor.hpp Makefile
	$(GPP) -g -c -U__STRICT_ANSI__ -o $@.mean.o -DEDGEBITS=28 -DPROOFSIZE=12 src/opencl/trimmer.cpp
#has rocm_smi_lib
#	$(GPP) -g -c -U__STRICT_ANSI__ -DPLATFORM_AMD -o $@.temp.o -DEDGEBITS=28 -DPROOFSIZE=12 src/opencl/miner.cpp -lrocm_smi64
#no rocm_smi_lib
	$(GPP) -g -c -U__STRICT_ANSI__ -o $@.temp.o -DEDGEBITS=28 -DPROOFSIZE=12 src/opencl/miner.cpp -lrocm_smi64
	ld -r -o $@ $@.temp.o $@.mean.o


libcudaminer.a: cudaN31L14.o blake2b.o siphash.o cuckoo.o
	ar cr $@ $^ $<.trimmer.o $<.miner.o 
	rm -rf cudaN31L14*.o
#	$(GPP) -g -o $@ $^ $<.temp.o  $<.mean.o -lOpenCL
#	ar cr $@ $^

libopenclminer.a: openclN31L14.o blake2b.o siphash.o cuckoo.o
	ar cr $@ $^ $<.temp.o  $<.mean.o
	rm -rf openclN31L14*.o

clean:
	rm -rf *.o *.a
