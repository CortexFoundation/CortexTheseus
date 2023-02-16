package filecache

import (
	"bytes"
	"io"
	"os"
	"sync"
	"time"
)

type cacheItem struct {
	content    []byte
	mutex      sync.RWMutex
	Size       int64
	Lastaccess time.Time
	Modified   time.Time
}

func (itm *cacheItem) WasModified(fi os.FileInfo) bool {
	itm.mutex.RLock()
	defer itm.mutex.RUnlock()
	return itm.Modified.Equal(fi.ModTime())
}

func (itm *cacheItem) GetReader() (b io.Reader) {
	b = bytes.NewReader(itm.Access())
	return
}

func (itm *cacheItem) Access() (c []byte) {
	itm.mutex.Lock()
	defer itm.mutex.Unlock()
	itm.Lastaccess = time.Now()
	c = itm.content
	return
}

func (itm *cacheItem) Dur() (t time.Duration) {
	itm.mutex.RLock()
	defer itm.mutex.RUnlock()
	t = time.Now().Sub(itm.Lastaccess)
	return
}
