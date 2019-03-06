#Detect OS
UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
OPENCL_HEADERS = "/opt/AMDAPPSDK-3.0/include"
LIBOPENCL = "/opt/amdgpu-pro/lib/x86_64-linux-gnu"
# LDLIBS = -lOpenCL
CC = gcc
endif

ifeq ($(UNAME), Darwin)
# Mac OS Frameworks
# OPENCL_HEADERS = "/System/Library/Frameworks/OpenCL.framework/Headers/"
# LIBOPENCL = "/System/Library/Frameworks/OpenCL.framework/Versions/Current/Libraries"
# LDLIBS = -framework OpenCL
# gcc installed with brew or macports cause xcode gcc is only clang wrapper
CC = g++
endif


CFLAGS += -std=c++11 -fPIC 
IDIR = include 
LDFLAG = -rdynamic -Llib 
LDFLAG += -Lcuckoo -lCuckoo -lblake2b -pthread

all: cuckoo libgominer.a libcuckoo.a

.PHONY: cuckoo
cuckoo:
	make -C cuckoo

%.o: %.cpp
	g++ -g -mavx2 -std=c++11 -c -fPIC $< -o $@

%.o: %.cc
	g++ -g -mavx2 -std=c++11 -c -fPIC $< -o $@

main: cuckoo/CuckooSolver.o minerBot.o main.cpp
	g++ --std=c++11 -o main -g  $^ -pthread -L./cuckoo -lblake2b

main-verify: cuckoo/siphash.o cuckoo/cuckoo.o gominer.o main.cpp 
	g++ --std=c++11 -DVERIFY_ONLY -o main-verify -g $^ -pthread -L./cuckoo -lblake2b 

libgominer.so: ${OBJ_LIB}
	g++ -fPIC -shared ${OBJ_LIB} -I${IDIR} ${LDFLAG} ${LDLIBS} -o ./libgominer.so

libcuckoo.a: gominer.o cuckoo/siphash.o cuckoo/cuckoo.o cuckoo/blake2b.o
	ar -rc libcuckoo.a $^

libgominer.a: minerBot.o
	ar -rc libgominer.a minerBot.o cuckoo/*.o

clean : cleanlib
	rm -f main *.o _temp_* util/*.o cuckoo/*.o

cleanlib:
	rm -f *.a cuckoo/*.a cuckoo/*.so

re : clean all
