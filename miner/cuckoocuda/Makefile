.POSIX:
.SUFFIXES:

OPT ?= -O3

GCC_ARCH_FLAGS ?= -march=native
GPP_ARCH_FLAGS ?= -march=native

# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS ?= -Wall -Wno-format -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L $(OPT) -DPREFETCH -I. $(CPPFLAGS) -pthread
GPP ?= g++ $(GPP_ARCH_FLAGS) -std=c++11 $(FLAGS)
CFLAGS ?= -Wall -Wno-format -fomit-frame-pointer $(OPT)
GCC ?= gcc $(GCC_ARCH_FLAGS) -std=gnu11 $(CFLAGS)

all : libgpugominer.a

blake2b.o: src/blake2.h src/blake2-impl.h src/blake2b-ref.c
	$(GCC) -c -o blake2b.o src/blake2b-ref.c

cudaN26L16.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=26 -DPROOFSIZE=16 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN31L22.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=31 -DPROOFSIZE=22 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN30L16.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=30 -DPROOFSIZE=16 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN29L16.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=29 -DPROOFSIZE=16 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN29L42.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=29 -DPROOFSIZE=42 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN28L42.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=28 -DPROOFSIZE=42 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN28L16.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=28 -DPROOFSIZE=16 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN25L16.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=25 -DPROOFSIZE=16 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN22L4.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=22 -DPROOFSIZE=4 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN24L4.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=24 -DPROOFSIZE=4 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN24L12.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=24 -DPROOFSIZE=12 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

cudaN28L8.o: src/siphash.cuh src/miner.cu Makefile
	nvcc -c -o $@.temp.o -rdc=true -DEDGEBITS=24 -DPROOFSIZE=4 -arch sm_35 src/miner.cu $(LIBS)
	nvcc -dlink -o $@ $@.temp.o -lcudart

libgpugominer.a:  cudaN24L12.o blake2b.o
	ar cr $@ $^ $<.temp.o 

clean:
	rm -rf *.o *.a

