make clean
make -j
cp libinterface.a int_mnist_model
cp src/interface.c int_mnist_model
cp src/interface.h int_mnist_model
cp src/test_interface.c int_mnist_model
cd int_mnist_model
./build.sh
