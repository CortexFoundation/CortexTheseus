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
### Rosseta
https://github.com/CortexFoundation/rosetta-cortex
### Docker
https://github.com/CortexFoundation/docker

## System Requirements
### **** x64 support  ****
```
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single pti ibrs ibpb stibp fsgsbase bmi1 avx2 smep bmi2 erms invpcid xsaveopt
```
For example
```
cat /proc/cpuinfo 
```
## Support
```
processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 63
model name	: Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz
stepping	: 2
microcode	: 0x1
cpu MHz		: 2494.224
cache size	: 30720 KB
physical id	: 0
siblings	: 2
core id		: 0
cpu cores	: 1
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl cpuid tsc_known_freq pni pclmulqdq ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm invpcid_single pti ibrs ibpb stibp fsgsbase bmi1 avx2 smep bmi2 erms invpcid xsaveopt
bugs		: cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf mds swapgs itlb_multihit
bogomips	: 4988.44
clflush size	: 64
cache_alignment	: 64
address sizes	: 46 bits physical, 48 bits virtual
```
## Not Support
``` 
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              32
On-line CPU(s) list: 0-31
Thread(s) per core:  2
Core(s) per socket:  16
Socket(s):           1
NUMA node(s):        2
Vendor ID:           AuthenticAMD
CPU family:          23
Model:               1
Model name:          AMD EPYC 7571
Stepping:            2
CPU MHz:             2534.021
BogoMIPS:            4399.86
Hypervisor vendor:   KVM
Virtualization type: full
L1d cache:           32K
L1i cache:           64K
L2 cache:            512K
L3 cache:            8192K
NUMA node0 CPU(s):   0-7,16-23
NUMA node1 CPU(s):   8-15,24-31
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid amd_dcm aperfmperf tsc_known_freq pni pclmulqdq ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm cmp_legacy cr8_legacy abm sse4a misalignsse 3dnowprefetch topoext perfctr_core vmmcall fsgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt sha_ni xsaveopt xsavec xgetbv1 clzero xsaveerptr arat npt nrip_save
```
### ubuntu
Cortex node is developed in Ubuntu 18.04 x64 + CUDA 9.2 + NVIDIA Driver 396.37 environment, with CUDA Compute capability >= 6.1. Latest Ubuntu distributions are also compatible, but not fully tested.
Recommend:
- cmake 3.11.0+
 ```
wget https://cmake.org/files/v3.11/cmake-3.11.0-rc4-Linux-x86_64.tar.gz
tar zxvf cmake-3.11.0-rc4-Linux-x86_64.tar.gz
sudo mv cmake-3.11.0-rc4-Linux-x86_64  /opt/cmake-3.11
sudo ln -sf /opt/cmake-3.11/bin/*  /usr/bin/

sudo apt-get install make
 ```
- go 1.20.+
```
wget https://go.dev/dl/go1.20.2.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.20.2.linux-amd64.tar.gz
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
### *centos (not recommended)
Recommend:
- cmake 3.11.0+
```
yum install cmake3
```
- go 1.17.+
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
### Running Testnet for developers (Bernard)
```
./cortex --bernard
```

