.POSIX:
.SUFFIXES:

OPT ?= -O3

GCC_ARCH_FLAGS ?= -march=native
GPP_ARCH_FLAGS ?= -march=native

# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS ?= -Wall -Wno-format -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L $(OPT) -DPREFETCH -I. -Isrc/opencl/ocl/include/ $(CPPFLAGS) -pthread -lOpenCL -g
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


cudaN31L14.o: src/siphash.cuh src/cuda/miner.cu src/cuda/wlt_trimmer.cu Makefile
	nvcc -g -std=c++11 -c -o $@.mean.o -rdc=true -DEDGEBITS=28 -DPROOFSIZE=12 -arch sm_35 src/cuda/wlt_trimmer.cu $(LIBS)
	nvcc -g -c -o $@.temp.o -rdc=true -DEDGEBITS=28 -DPROOFSIZE=12 -arch sm_35 src/cuda/miner.cu $(LIBS)
	nvcc -g -ccbin g++ -std=c++11 -m64 -Xcompiler -fpermissive -gencode arch=compute_35,code=sm_35 -rdc=true -dlink -o $@ $@.temp.o $@.mean.o 

openclN31L14.o: src/siphash.cuh src/opencl/miner_cl.cpp src/opencl/wlt_trimmer.cpp Makefile
	$(GPP) -g -c -U__STRICT_ANSI__ -o $@.mean.o -DEDGEBITS=28 -DPROOFSIZE=12 src/opencl/wlt_trimmer.cpp
	$(GPP) -g -c -U__STRICT_ANSI__ -o $@.temp.o -DEDGEBITS=28 -DPROOFSIZE=12 src/opencl/miner_cl.cpp
	ld -r -o $@ $@.temp.o $@.mean.o

libcudaminer.a: cudaN31L14.o blake2b.o siphash.o cuckoo.o
	ar cr $@ $^ $<.temp.o  $<.mean.o
	rm -rf cudaN31L14.o cudaN31L14.o.temp.o $<.temp.p $<.mean.o
#	$(GPP) -g -o $@ $^ $<.temp.o  $<.mean.o -lOpenCL
#	ar cr $@ $^

libopenclminer.a: openclN31L14.o blake2b.o siphash.o cuckoo.o
	ar cr $@ $^ $<.temp.o  $<.mean.o
	rm -rf $^ $<.temp.o $<.mean.o
clean:
	rm -rf *.o *.a

