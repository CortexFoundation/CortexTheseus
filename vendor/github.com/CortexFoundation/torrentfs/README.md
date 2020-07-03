# P2P file system of cortex full node
```
go get github.com/CortexFoundation/torrentfs

or

make
```
#### torrent-create : to create torrent file
```./torrent-create file > torrent```
#### torrent-magnet : load info hash from torrent file
```./torrent-magnet < torrent```
#### seeding : to seed file to dht
```./seeding -dataDir=store```
under store folder
```
ec6b1f5b5073c07dd35a53a3a13220c1a21e426d
├── data
└── torrent
```
#### torrent : to download file
```./torrent download $magnet```
