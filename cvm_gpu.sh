export LD_LIBRARY_PATH=$PWD/plugins:$LD_LIBRARY_PATH
./build/bin/cortex cvm --infer.devicetype=gpu
