#Detect OS
UNAME := $(shell uname)

ifeq ($(UNAME), Linux)
OPENCL_HEADERS = "/opt/AMDAPPSDK-3.0/include"
LIBOPENCL = "/opt/amdgpu-pro/lib/x86_64-linux-gnu"
LDLIBS = -lOpenCL
CC = gcc
endif

ifeq ($(UNAME), Darwin)
# Mac OS Frameworks
OPENCL_HEADERS = "/System/Library/Frameworks/OpenCL.framework/Headers/"
LIBOPENCL = "/System/Library/Frameworks/OpenCL.framework/Versions/Current/Libraries"
LDLIBS = -framework OpenCL
# gcc installed with brew or macports cause xcode gcc is only clang wrapper
CC = g++
endif


CXXFLAGS += -std=c++11
IDIR = include 
LDFLAG = -rdynamic -Llib -ltinyxml2  -L${LIBOPENCL}
LDFLAG += -Lzcash -lequiSolver -Lcuckoo -lCuckoo -lblake2b


#INCLUDES = minerBot.h \
#zcash/blake.h zcash/sha256.h zcash/equihash.hpp zcash/cl_helper.h
#OBJ = main.o minerBot.o util/XMLHelper.o\
#zcash/blake.o zcash/sha256.o zcash/equihash.o

INCLUDES = minerBot.h
OBJ = main.o minerBot.o util/XMLHelper.o
OBJ_LIB = minerBot.o util/XMLHelper.o

all: main

${OBJ}: ${INCLUDES}
${OBJ_LIB}: ${INCLUDES}

main: ${OBJ}
	g++ -o main ${OBJ} -I${IDIR} ${LDFLAG} ${LDLIBS}

lib: ${OBJ_LIB}
	g++ -fPIC -shared ${OBJ_LIB}  -I${IDIR} ${LDFLAG} ${LDLIBS} -o ./go_interface/libgominer.so

clean :
	rm -f main *.o _temp_*

re : clean all