## Related projects
### CVM runtime (AI container)
https://github.com/CortexFoundation/cvm-runtime
### File storage
Stop your cortex full node daemon, when you do this test

https://github.com/CortexFoundation/torrentfs
```
git clone https://github.com/CortexFoundation/torrentfs.git
cd torrentfs
make
./build/bin/torrent download 'ih:6b75cc1354495ec763a6b295ee407ea864a0c292'
./build/bin/torrent download 'ih:b2f5b0036877be22c6101bdfa5f2c7927fc35ef8'
./build/bin/torrent download 'ih:5a49fed84aaf368cbf472cc06e42f93a93d92db5'
./build/bin/torrent download 'ih:1f1706fa53ce0723ba1c577418b222acbfa5a200'
./build/bin/torrent download 'ih:3f1f6c007e8da3e16f7c3378a20a746e70f1c2b0'
```
downloaded ALL the torrents !!!!!!!!!!!!!!!!!!!

##### *** Make sure you can download the file successfully
##### *** Accept in/out traffic of fw settings as possible for stable and fast downloading speed
##### (40401 40404 5008 both in and out(tcp udp) traffic accepted at least)

### AI wrapper (Fixed API for inference and file storage)
https://github.com/CortexFoundation/inference
### PoW (Cortex Cuckoo cycle)
https://github.com/CortexFoundation/solution

## System Requirements
### ubuntu
Cortex node is developed in Ubuntu 18.04 x64 + CUDA 9.2 + NVIDIA Driver 396.37 environment, with CUDA Compute capability >= 6.1. Latest Ubuntu distributions are also compatible, but not fully tested.
Recommend:
- cmake 3.11.0+
 ```
wget https://cmake.org/files/v3.11/cmake-3.11.0-rc4-Linux-x86_64.tar.gz
tar zxvf cmake-3.11.0-rc4-Linux-x86_64.tar.gz
sudo mv cmake-3.11.0-rc4-Linux-x86_64  /opt/cmake-3.11
sudo ln -sf /opt/cmake-3.11/bin/*  /usr/bin/
 ```
- go 1.16+
```
wget https://dl.google.com/go/go1.16.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.16.linux-amd64.tar.gz
echo 'export PATH="$PATH:/usr/local/go/bin"' >> ~/.bashrc
source ~/.bashrc
```
- gcc/g++ 5.4+
```
sudo apt install gcc
sudo apt install g++
```
- cuda 9.2+ (if u have gpu)
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
```
- nvidia driver 396.37+ reference: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#major-components
- ubuntu 18.04+
### centos
Recommend:
- cmake 3.11.0+
```
yum install cmake3
```
- go 1.15.5+
- gcc/g++ 5.4+ reference: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements
```
sudo yum install centos-release-scl
sudo yum install devtoolset-7-gcc*
scl enable devtoolset-7 bash
which gcc
gcc --version
```
- cuda 10.1+ (if u have gpu)
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
```
- nvidia driver 418.67+
- centos 7.6

## Cortex Full Node

### Compile Source Code (8G+ Memory suggested)
1. git clone --recursive https://github.com/CortexFoundation/CortexTheseus.git
2. cd CortexTheseus
3. make clean && make -j$(nproc)

### It is important to pass this check of libcvm_runtime.so
ldd plugins/libcvm_runtime.so
```
linux-vdso.so.1 =>  (0x00007ffe107fa000)
libstdc++.so.6 => /lib64/libstdc++.so.6 (0x00007f250e6a8000)
libm.so.6 => /lib64/libm.so.6 (0x00007f250e3a6000)
libgomp.so.1 => /lib64/libgomp.so.1 (0x00007f250e180000)
libgcc_s.so.1 => /lib64/libgcc_s.so.1 (0x00007f250df6a000)
libpthread.so.0 => /lib64/libpthread.so.0 (0x00007f250dd4e000)
libc.so.6 => /lib64/libc.so.6 (0x00007f250d980000)
/lib64/ld-linux-x86-64.so.2 (0x00007f250ed35000)
```

(If failed, run ```rm -rf cvm-runtime && git submodule init && git submodule update``` and try again)

### Running Bash

And then, run any command to start full node `cortex`:

```Bash
1. cd CortexTheseus
2. export LD_LIBRARY_PATH=$PWD:$PWD/plugins:$LD_LIBRARY_PATH
3. ./build/bin/cortex

It is easy for you to view the help document by running ./build/bin/cortex --help
```
