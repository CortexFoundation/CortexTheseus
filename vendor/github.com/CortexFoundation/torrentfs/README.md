# P2P file system of cortex full node

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
