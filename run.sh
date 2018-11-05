make clean -C miner/cuckoocuda/
make -C miner/cuckoocuda/
make clean && make
cp miner/cuckoocuda/src/wlt_trimmer.cl ./
./build/bin/miner --account "0xe291d43ad2eb6ea04e9f5e1a0c67f970702f8bd6" --deviceid 1 --devicetype gpu --pool_uri 192.168.50.75:8018 --verbosity 5

