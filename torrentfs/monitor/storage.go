package monitor

import (
	"io/ioutil"
	"log"
	"os"
	"path"
)

const (
	defaultTorrentName = "torrent"
)

// InitStorage ...
func InitStorage(storageDir string, manager *TorrentManager) {
	files, err := ioutil.ReadDir(storageDir)
	if err != nil {
		log.Fatal(err)
	}
	for _, f := range files {
		infohash := f.Name()
		if f.IsDir() && len(infohash) == 40 {
			torrentPath := path.Join(storageDir, f.Name(), defaultTorrentName)
			if _, err := os.Stat(torrentPath); err == nil {
				manager.NewTorrent(torrentPath)
			} else {
				mURI := "magnet:?xt=urn:btih:" + infohash
				manager.NewTorrent(mURI)
			}
		}
	}
}
