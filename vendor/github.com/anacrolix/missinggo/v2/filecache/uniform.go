package filecache

import (
	"io"
	"os"
	"strings"

	"github.com/anacrolix/missinggo/v2/resource"
)

type uniformResourceProvider struct {
	*Cache
}

var _ resource.Provider = &uniformResourceProvider{}

func (me *uniformResourceProvider) NewInstance(loc string) (resource.Instance, error) {
	return &uniformResource{me.Cache, loc}, nil
}

type uniformResource struct {
	Cache    *Cache
	Location string
}

func (me *uniformResource) Get() (io.ReadCloser, error) {
	return me.Cache.OpenFile(me.Location, os.O_RDONLY)
}

func (me *uniformResource) Put(r io.Reader) (err error) {
	f, err := me.Cache.OpenFile(me.Location, os.O_WRONLY|os.O_CREATE|os.O_TRUNC)
	if err != nil {
		return
	}
	defer f.Close()
	_, err = io.Copy(f, r)
	return
}

func (me *uniformResource) ReadAt(b []byte, off int64) (n int, err error) {
	f, err := me.Cache.OpenFile(me.Location, os.O_RDONLY)
	if err != nil {
		return
	}
	defer f.Close()
	return f.ReadAt(b, off)
}

func (me *uniformResource) WriteAt(b []byte, off int64) (n int, err error) {
	f, err := me.Cache.OpenFile(me.Location, os.O_CREATE|os.O_WRONLY)
	if err != nil {
		return
	}
	defer f.Close()
	return f.WriteAt(b, off)
}

func (me *uniformResource) Stat() (fi os.FileInfo, err error) {
	return me.Cache.Stat(me.Location)
}

func (me *uniformResource) Delete() error {
	return me.Cache.Remove(me.Location)
}

func (me *uniformResource) Readdirnames() (names []string, err error) {
	prefix := me.Location + "/"
	me.Cache.WalkItems(func(info ItemInfo) {
		//log.Printf("%q %q", me.Location, info.Path)
		name := string(info.Path)
		if strings.HasPrefix(name, prefix) {
			names = append(names, name[len(prefix):])
		}
	})
	return
}
