package torrentfs

import (
	 "fmt"
	"io/ioutil"
	_ "github.com/CortexFoundation/CortexTheseus/log"
	_ "github.com/CortexFoundation/CortexTheseus/p2p"
	_ "github.com/CortexFoundation/CortexTheseus/params"
	_ "github.com/CortexFoundation/CortexTheseus/rpc"
)

type CVMStorage interface {
	Available(infohash string, rawSize int64) bool
	Exist(infohash string) bool
	GetFile(infohash string, path string) ([]byte, error)
	ExistTorrent(infohash string) bool
}

func CreateStorage(storage_type string, config Config) CVMStorage {
	if storage_type == "simple" {
		return &InfoHashFileSystem{
			DataDir: config.DataDir,
		}
	} else if storage_type == "torrent" {
		return &TorrentFS{
			config: &config,
		}
	}
	return nil
}

type InfoHashFileSystem struct {
	DataDir   string
}

func (fs InfoHashFileSystem) Available(infohash string, rawSize int64) bool {
	// modelDir := fs.DataDir + "/" + infoHash
	// if (os.Stat)
	return true
}

func (fs InfoHashFileSystem) Exist(infohash string) bool {
	return true
}

func (fs InfoHashFileSystem) GetFile(infohash string, path string) ([]byte, error) {
	fn := fs.DataDir + "/" + infohash  + "/" + path
	data, err := ioutil.ReadFile(fn)
	fmt.Println("InfoHashFileSystem", "GetFile", fn)
	return data, err

}
func (fs InfoHashFileSystem) ExistTorrent(infohash string) bool {

	return false
}
