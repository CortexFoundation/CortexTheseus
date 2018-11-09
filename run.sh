make clean -C miner/libcuckoo/
make -C miner/libcuckoo/
make clean && make
cp miner/libcuckoo/src/opencl/wlt_trimmer.cl ./
./build/bin/cuda_miner --account "0xe291d43ad2eb6ea04e9f5e1a0c67f970702f8bd6" --deviceid 0 --devicetype gpu --pool_uri 192.168.50.75:8009 --verbosity 5

