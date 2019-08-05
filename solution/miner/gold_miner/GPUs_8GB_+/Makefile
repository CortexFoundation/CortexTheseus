# Target rules
all: build

build: cuckoo

cuckoo: cuckoo.o
	nvcc -ccbin g++ -std=c++11  -m64  -Xcompiler -fpermissive -gencode arch=compute_35,code=sm_35 -rdc=true -o cuckoo cuckoo.o
cuckoo.o: kernel.cu
	nvcc -std=c++11 -c -o cuckoo.o -rdc=true -DEDGEBITS=24 -DPROOFSIZE=4 -arch sm_35 kernel.cu
clean:
	rm -f cuckoo cuckoo.o
