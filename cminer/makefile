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
# LDFLAG += -Lzcash -lequiSolver -L${LIBOPENCL}
LDFLAG += -Lcuckoo -lCuckoo -lblake2b -pthread


#INCLUDES = minerBot.h \
#zcash/blake.h zcash/sha256.h zcash/equihash.hpp zcash/cl_helper.h
#OBJ = main.o minerBot.o util/XMLHelper.o\
#zcash/blake.o zcash/sha256.o zcash/equihash.o

INCLUDES = minerBot.h
OBJ = main.o minerBot.o
OBJ_LIB = minerBot.o

all: cuckoo libgominer.a

# ${OBJ}: ${INCLUDES}
# ${OBJ_LIB}: ${INCLUDES}

.PHONY: cuckoo
cuckoo:
	make -C cuckoo

%.o: %.cpp
	g++ -g -mavx2 -std=c++11 -c -fPIC $< -o $@

main: cuckoo/CuckooSolver.o minerBot.o main.cpp
	g++ --std=c++11 -o main -g  $^ -pthread -L./cuckoo -lblake2b

main-verify:cuckoo/siphash.o cuckoo/cuckoo.o  main.cpp 
	g++ --std=c++11 -DVERIFY_ONLY -o main-verify -g $^ -pthread -L./cuckoo -lblake2b 

libgominer.so: ${OBJ_LIB}
	g++ -fPIC -shared ${OBJ_LIB}  -I${IDIR} ${LDFLAG} ${LDLIBS} -o ./libgominer.so

libgominer.a: $(OBJ_LIB)
	ar -rc libgominer.a $(OBJ_LIB) cuckoo/*.o

clean : cleanlib
	rm -f main *.o _temp_* util/*.o cuckoo/*.o

cleanlib:
	rm -f *.a cuckoo/*.a cuckoo/*.so

re : clean all
