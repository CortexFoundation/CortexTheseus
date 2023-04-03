package filecache

import (
	"bytes"
	"io"
	"os"
	"sync"
	"time"
)

type cacheItem struct {
	name        string
	content     []byte
	mutex       sync.RWMutex
	Size        int64
	Lastaccess  time.Time
	Modified    time.Time
	AccessCount uint64
}

func (itm *cacheItem) Key() string {
	return itm.name
}

func (itm *cacheItem) Content() []byte {
	return itm.content
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
	itm.AccessCount++
	c = itm.content
	return
}

func (itm *cacheItem) Dur() (t time.Duration) {
	itm.mutex.RLock()
	defer itm.mutex.RUnlock()
	t = time.Now().Sub(itm.Lastaccess)
	return
}

// CacheItemPair maps key to access counter
type CacheItemPair struct {
	Key         string
	AccessCount uint64
}

type CacheItemPairList []CacheItemPair

func (p CacheItemPairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p CacheItemPairList) Len() int           { return len(p) }
func (p CacheItemPairList) Less(i, j int) bool { return p[i].AccessCount > p[j].AccessCount }
