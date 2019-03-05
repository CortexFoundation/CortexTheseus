#make clean -C miner/libcuckoo/
#make -C miner/libcuckoo/
#make clean && make cuda-miner
#./build/bin/cuda_miner --account "0xe291d43ad2eb6ea04e9f5e1a0c67f970702f8bd6" --deviceids 0,1 --pool_uri miner.cortexlabs.ai:8009 --verbosity 5


make clean -C miner/libcuckoo && make clean && make opencl-miner && ./build/bin/opencl_miner

