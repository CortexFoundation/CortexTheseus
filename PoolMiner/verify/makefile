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


CFLAGS += -fPIC 
IDIR = include 

all: libcuckoo.a

%.o: %.cpp
	g++ -g -mavx2 -std=c++11 -c -fPIC $< -o $@

%.o: %.cc
	g++ -g -mavx2 -std=c++11 -c -fPIC $< -o $@

libcuckoo.a: gominer.o siphash.o cuckoo.o blake2b.o
	ar -rc libcuckoo.a $^

clean : cleanlib
	rm -f *.o _temp_* util/*.o cuckoo/*.o

cleanlib:
	rm -f *.a cuckoo/*.a cuckoo/*.so

re : clean all
