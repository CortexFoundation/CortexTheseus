package filecache

import (
	"errors"
	"os"
	"sync"

	"github.com/anacrolix/missinggo/v2/pproffd"
)

type File struct {
	path       key
	f          pproffd.OSFile
	afterWrite func(endOff int64)
	onRead     func(n int)
	mu         sync.Mutex
	offset     int64
}

func (me *File) Seek(offset int64, whence int) (ret int64, err error) {
	ret, err = me.f.Seek(offset, whence)
	if err != nil {
		return
	}
	me.offset = ret
	return
}

var (
	ErrFileTooLarge    = errors.New("file too large for cache")
	ErrFileDisappeared = errors.New("file disappeared")
)

func (me *File) Write(b []byte) (n int, err error) {
	n, err = me.f.Write(b)
	me.offset += int64(n)
	me.afterWrite(me.offset)
	return
}

func (me *File) WriteAt(b []byte, off int64) (n int, err error) {
	n, err = me.f.WriteAt(b, off)
	me.afterWrite(off + int64(n))
	return
}

func (me *File) Close() error {
	return me.f.Close()
}

func (me *File) Stat() (os.FileInfo, error) {
	return me.f.Stat()
}

func (me *File) Read(b []byte) (n int, err error) {
	n, err = me.f.Read(b)
	me.onRead(n)
	return
}

func (me *File) ReadAt(b []byte, off int64) (n int, err error) {
	n, err = me.f.ReadAt(b, off)
	me.onRead(n)
	return
}
