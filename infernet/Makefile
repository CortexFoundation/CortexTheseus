ROOTDIR = $(CURDIR)

.PHONY: clean all test

INCLUDE_FLAGS = -Iinclude
PKG_CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
PKG_LDFLAGS =

all: cpu gpu

cpu:
		@mkdir -p build/cpu && cd build/cpu && cmake ../.. -DUSE_CUDA=OFF && $(MAKE)

gpu:
		@mkdir -p build/gpu && cd build/gpu && cmake ../.. && $(MAKE)

clean:
		@mkdir -p build/gpu && cd build/gpu && cmake ../.. && $(MAKE) clean
		@mkdir -p build/cpu && cd build/cpu && cmake ../.. && $(MAKE) clean
