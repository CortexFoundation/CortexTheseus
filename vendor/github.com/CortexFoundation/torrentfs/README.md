# Torrentfs

Torrentfs is a elastic file system, you can pull or push any files by following torrentfs protocol anywhere, anytime

## Install (ubuntu & centos)
```
go install github.com/CortexFoundation/torrentfs/cmd/torrent@latest
```

## Import
```
go get github.com/CortexFoundation/torrentfs
```
## How to use
```
https://github.com/CortexFoundation/torrentfs.git
cd torrentfs
make
```
#### Create torrent file by 4k pieces
```
./build/bin/torrent-create workspace/data -p=4096 > workspace/test-torrent
```
#### Load info hash from torrent file
```
./build/bin/torrent-magnet < workspace/test-torrent
```
```
magnet:?xt=urn:btih:9196320d998fdab966bcb3a08f3f087e1f993c12&dn=data&tr=udp%3A%2F%2Ftracker.cortexlabs.ai%3A5008
```
#### Seed file to dht
```
mkdir -p mnt/9196320d998fdab966bcb3a08f3f087e1f993c12/data
cp workspace/test-torrent mnt/9196320d998fdab966bcb3a08f3f087e1f993c12/torrent
cp -r workspace/data/* mnt/9196320d998fdab966bcb3a08f3f087e1f993c12/data
./build/bin/seeding -dataDir=mnt
```
#### Download file
```
./build/bin/torrent download 'infohash:9196320d998fdab966bcb3a08f3f087e1f993c12'
```
#### How to test your network for torrent ?
```
./build/bin/torrent download 'infohash:6b75cc1354495ec763a6b295ee407ea864a0c292'
./build/bin/torrent download 'infohash:b2f5b0036877be22c6101bdfa5f2c7927fc35ef8'
./build/bin/torrent download 'infohash:5a49fed84aaf368cbf472cc06e42f93a93d92db5'
./build/bin/torrent download 'infohash:1f1706fa53ce0723ba1c577418b222acbfa5a200'
./build/bin/torrent download 'infohash:3f1f6c007e8da3e16f7c3378a20a746e70f1c2b0'
```
downloaded ALL the torrents !!!!!!!!!!!!!!!!!!!
## Server running
https://github.com/CortexFoundation/torrentfs/pull/216
## Seeding or Sharing
https://github.com/CortexFoundation/torrentfs/pull/224
# Special thanks

[Anacrolix BitTorrent client package and utilities](https://github.com/anacrolix/torrent)
