## Torrentfs

Torrentfs is a elastic file system, you can pull or push any files by following torrentfs protocol anywhere, anytime

## Server running
https://github.com/CortexFoundation/torrentfs/pull/216
## Seeding or Sharing
https://github.com/CortexFoundation/torrentfs/pull/224

# P2P file system of cortex full node
![image](https://user-images.githubusercontent.com/22344498/118778205-6ef75f00-b8bc-11eb-880e-17b5bea66814.png)


## Import
```
go get github.com/CortexFoundation/torrentfs
```
## Build
```
https://github.com/CortexFoundation/torrentfs.git
cd torrentfs
make
```
#### How to test your network for torrent ?
```
./build/bin/torrent download 'ih:6b75cc1354495ec763a6b295ee407ea864a0c292'
./build/bin/torrent download 'ih:b2f5b0036877be22c6101bdfa5f2c7927fc35ef8'
./build/bin/torrent download 'ih:5a49fed84aaf368cbf472cc06e42f93a93d92db5'
./build/bin/torrent download 'ih:1f1706fa53ce0723ba1c577418b222acbfa5a200'
./build/bin/torrent download 'ih:3f1f6c007e8da3e16f7c3378a20a746e70f1c2b0'
```
downloaded ALL the torrents !!!!!!!!!!!!!!!!!!!

## How to use
```
cd torrentfs
make
```
#### Create torrent file
```
./build/bin/torrent-create test/data -p=4096 > test-torrent
```
#### Load info hash from torrent file
```
./build/bin/torrent-magnet < test-torrent
```
magnet:?xt=urn:btih:91b709fdeaadd7dbb3c798fad493ac73e1c60d7c&dn=data&tr=udp%3A%2F%2Ftracker.cortexlabs.ai%3A5008
#### Seed file to dht
```
mkdir -p mnt/91b709fdeaadd7dbb3c798fad493ac73e1c60d7c/data
cp test-torrent mnt/91b709fdeaadd7dbb3c798fad493ac73e1c60d7c/torrent
cp -r testdata/data/* mnt/91b709fdeaadd7dbb3c798fad493ac73e1c60d7c/data
./build/bin/seeding -dataDir=mnt
```
#### Download file
```
./build/bin/torrent download 'infohash:91b709fdeaadd7dbb3c798fad493ac73e1c60d7c'
```
# Special thanks

[Anacrolix BitTorrent client package and utilities](https://github.com/anacrolix/torrent)
