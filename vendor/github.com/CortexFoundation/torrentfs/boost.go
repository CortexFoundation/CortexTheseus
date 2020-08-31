// Copyright 2020 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

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
		link := node + "/" + ih + "/" + name
		ret, err := f.getFileFromURI(link)
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
		ret, err := f.getFileFromURI(node + "/" + ih + "/torrent")
		if err == nil {
			buf := bytes.NewBuffer(ret)
			mi, err2 := metainfo.Load(buf)
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

func (f *BoostDataFetcher) FetchFile(ih, subpath string) ([]byte, error) {
	return f.getFile(ih, subpath)
}

func (f *BoostDataFetcher) FetchTorrent(ih string) ([]byte, error) {
	return f.getTorrentFromBoostNodes(f.nodes, ih)
}
