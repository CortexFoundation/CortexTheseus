package torrentfs

import "bytes"
import "errors"
import "net/http"
import "time"

//import "fmt"
import "io/ioutil"
import "github.com/anacrolix/torrent/metainfo"

var Str404NotFound []byte = []byte{60, 104, 116, 109, 108, 62, 13, 10, 60, 104, 101, 97, 100, 62, 60, 116, 105, 116, 108, 101, 62, 52, 48, 52, 32, 78, 111, 116, 32, 70, 111, 117, 110, 100, 60, 47, 116, 105, 116, 108, 101, 62, 60, 47, 104, 101, 97, 100, 62, 13, 10, 60, 98, 111, 100, 121, 32, 98, 103, 99, 111, 108, 111, 114, 61, 34, 119, 104, 105, 116, 101, 34, 62, 13, 10, 60, 99, 101, 110, 116, 101, 114, 62, 60, 104, 49, 62, 52, 48, 52, 32, 78, 111, 116, 32, 70, 111, 117, 110, 100, 60, 47, 104, 49, 62, 60, 47, 99, 101, 110, 116, 101, 114, 62, 13, 10, 60, 104, 114, 62, 60, 99, 101, 110, 116, 101, 114, 62, 110, 103, 105, 110, 120, 47, 49, 46, 49, 52, 46, 48, 32, 40, 85, 98, 117, 110, 116, 117, 41, 60, 47, 99, 101, 110, 116, 101, 114, 62, 13, 10, 60, 47, 98, 111, 100, 121, 62, 13, 10, 60, 47, 104, 116, 109, 108, 62, 13, 10}

func (f *BoostDataFetcher) getFileFromURI(uri string) ([]byte, error) {
	var client = http.Client{
		Timeout: 30 * time.Second,
	}
	resp, err := client.Get(uri)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	ret, err := ioutil.ReadAll(resp.Body)
	if err != nil || bytes.HasPrefix(ret, Str404NotFound) {
		return nil, errors.New("404 Not Found")
	}
	return ret, nil
}

type BoostDataFetcher struct {
	nodes []string
}

func NewBoostDataFetcher(nodes []string) *BoostDataFetcher {
	return &BoostDataFetcher{
		nodes: nodes,
	}
}

func (f *BoostDataFetcher) getFileFromBoostNodes(nodes []string, ih, name string) ([]byte, error) {
	for _, node := range nodes {
		ret, err := f.getFileFromURI(node + "/files/" + ih + "/" + name)
		if err == nil {
			return ret, nil
		}
	}
	return nil, errors.New("404 Not Found")
}

func (f *BoostDataFetcher) getFile(ih, name string) ([]byte, error) {
	return f.getFileFromBoostNodes(f.nodes, ih, name)
}

func (f *BoostDataFetcher) getTorrentFromBoostNodes(nodes []string, ih string) ([]byte, error) {
	for _, node := range nodes {
		ret, err := f.getFileFromURI(node + "/files/" + ih + "/torrent")
		if err == nil {
			mi, err2 := metainfo.Load(bytes.NewBuffer(ret))
			if err2 != nil {
				continue
			}
			ih2 := metainfo.HashBytes(mi.InfoBytes)
			if ih != ih2.String() {
				continue
			}
			return ret, nil
		}
	}
	return nil, errors.New("Torrent Not Found")
}

func (f *BoostDataFetcher) GetFile(ih, subpath string) ([]byte, error) {
	return f.getFile(ih, subpath)
}

func (f *BoostDataFetcher) GetTorrent(ih string) ([]byte, error) {
	return f.getTorrentFromBoostNodes(f.nodes, ih)
}
