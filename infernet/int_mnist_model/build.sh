nvcc test_interface.c -linterface -L. -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn -o test_interface

