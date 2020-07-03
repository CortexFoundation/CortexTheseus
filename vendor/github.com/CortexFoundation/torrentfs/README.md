# P2P file system of cortex full node
```
go get github.com/CortexFoundation/torrentfs

or

make
```
```cd build/bin```

```mkdir -p store/0803be8fc7309d155dfcee65a92a6254bd55a3d2```

```echo "Hello torrent" > store/0803be8fc7309d155dfcee65a92a6254bd55a3d2/data ```

#### torrent-create : to create torrent file
```./torrent-create store/0803be8fc7309d155dfcee65a92a6254bd55a3d2/data > store/0803be8fc7309d155dfcee65a92a6254bd55a3d2/torrent```
#### torrent-magnet : load info hash from torrent file
```./torrent-magnet < store/0803be8fc7309d155dfcee65a92a6254bd55a3d2/torrent```

```
tree store

0803be8fc7309d155dfcee65a92a6254bd55a3d2
├── data
└── torrent
```
#### seeding : to seed file to dht
```./seeding -dataDir=store```
#### torrent : to download file
```./torrent download 'infohash:0803be8fc7309d155dfcee65a92a6254bd55a3d2' ```
