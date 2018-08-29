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


# Change this path if the SDK was installed in a non-standard location
# By default libOpenCL.so is searched in default system locations, this path
# lets you adds one more directory to the search path.

IDIR = ../include 
LDFLAG = -ltinyxml2 -rdynamic -L${LIBOPENCL}
CXXFLAGS += -std=c++11

INCLUDES = blake.h sha256.h equihash.hpp cl_helper.h _kernel.h

OBJ = blake.o sha256.o equihash.o

INCLUDES = blake.h param.h _kernel.h sha256.h

all : lib

#sa-solver : ${OBJ}
#	${CC} -o sa-solver ${OBJ} ${LDFLAGS} ${LDLIBS} ${CFLAGS}

lib : ${OBJ}
	ar rcs libequiSolver.a ${OBJ}

${OBJ} : ${INCLUDES}

#blake.o: ${INCLUDES}
#	g++ -c blake.c ${CPPFLAG}

#sha256.o: ${INCLUDES}
#	g++ -c sha256.c ${CPPFLAG}

#equihash.o: ${INCLUDES}
#	g++ -c equihash.cpp ${CPPFLAG}

_kernel.h : input.cl param.h
	echo 'const char *ocl_code = R"_mrb_(' >$@
	cpp $< >>$@
	echo ')_mrb_";' >>$@

#test : sa-solver
#	./sa-solver --nonces 100 -v -v 2>&1 | grep Soln: | \
#	    diff -u testing/sols-100 - | cut -c 1-75

clean :
	rm -f libequiSolver.a _kernel.h *.o _temp_*

re : clean all
