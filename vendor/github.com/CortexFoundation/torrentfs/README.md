# Independent HTTP Service for file seeding
https://github.com/CortexFoundation/torrentfs/pull/216

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
cd build/bin
mkdir -p mnt/0803be8fc7309d155dfcee65a92a6254bd55a3d2
echo "Hello torrent" > mnt/0803be8fc7309d155dfcee65a92a6254bd55a3d2/data
```
#### Create torrent file
```
./torrent-create mnt/0803be8fc7309d155dfcee65a92a6254bd55a3d2/data > mnt/0803be8fc7309d155dfcee65a92a6254bd55a3d2/torrent
```
#### Load info hash from torrent file
```
./torrent-magnet < mnt/0803be8fc7309d155dfcee65a92a6254bd55a3d2/torrent
tree mnt
0803be8fc7309d155dfcee65a92a6254bd55a3d2
├── data
└── torrent
```
#### Seed file to dht
```./seeding -dataDir=mnt```
#### Download file
```
./torrent download 'infohash:0803be8fc7309d155dfcee65a92a6254bd55a3d2'
ls -alt data && md5sum data && cat data
```
# Special thanks

[Anacrolix BitTorrent client package and utilities](https://github.com/anacrolix/torrent)
