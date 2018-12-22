nvcc test_interface.c -linterface -L. -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn -o test_interface

nvcc test_leak.c -lcortexnet -L. -I../include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn -o test_leak
