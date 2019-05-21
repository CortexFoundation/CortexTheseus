ROOTDIR = $(CURDIR)

.PHONY: clean all test

INCLUDE_FLAGS = -Iinclude
PKG_CFLAGS = -std=c++11 -Wall -O2 $(INCLUDE_FLAGS) -fPIC
PKG_LDFLAGS =

all:
	@mkdir -p build && cd build && cmake .. && $(MAKE)

runtime:
	@mkdir -p build && cd build && cmake .. && $(MAKE) runtime

# clean rule
clean:
	@mkdir -p build && cd build && cmake .. && $(MAKE) clean
