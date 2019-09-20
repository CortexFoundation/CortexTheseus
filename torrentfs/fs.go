package torrentfs

import (
	"fmt"
	_ "github.com/CortexFoundation/CortexTheseus/log"
	_ "github.com/CortexFoundation/CortexTheseus/p2p"
	_ "github.com/CortexFoundation/CortexTheseus/params"
	_ "github.com/CortexFoundation/CortexTheseus/rpc"
	"io/ioutil"
	"path"
)

type CVMStorage interface {
	Available(infohash string, rawSize int64) (bool, error)
	GetFile(infohash string, path string) ([]byte, error)
	Stop() error
}

func CreateStorage(storage_type string, config Config) CVMStorage {
	if storage_type == "simple" {
		return &InfoHashFileSystem{
			DataDir: config.DataDir,
		}
	} else if storage_type == "torrent" {
		return GetTorrentInstance()
	}
	return nil
}

type InfoHashFileSystem struct {
	DataDir string
}

func (fs *InfoHashFileSystem) Available(infohash string, rawSize int64) (bool, error) {
	return true, nil
}

func (fs *InfoHashFileSystem) Stop() error {
	return nil
}

func (fs *InfoHashFileSystem) GetFile(infohash string, subpath string) ([]byte, error) {
	fn := path.Join(fs.DataDir, infohash, subpath)
	data, err := ioutil.ReadFile(fn)
	fmt.Println("InfoHashFileSystem", "GetFile", fn, "err", err)
	return data, err
}
