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
	ModTime     time.Time
	AccessCount uint64
}

// Key returns the name of the cached item.
func (itm *cacheItem) Key() string {
	return itm.name
}

// Content returns the cached content.
func (itm *cacheItem) Content() []byte {
	itm.mutex.RLock()
	defer itm.mutex.RUnlock()
	return itm.content
}

// WasModified checks if the file has the same mod time as when cached.
func (itm *cacheItem) WasModified(fi os.FileInfo) bool {
	itm.mutex.RLock()
	defer itm.mutex.RUnlock()
	return itm.ModTime.Equal(fi.ModTime())
}

// GetReader returns a new reader for the cached content.
func (itm *cacheItem) GetReader() io.Reader {
	itm.mutex.RLock()
	defer itm.mutex.RUnlock()
	return bytes.NewReader(itm.content)
}

// Access returns the cached content and updates last access time and access count.
func (itm *cacheItem) Access() []byte {
	itm.mutex.Lock()
	itm.Lastaccess = time.Now()
	itm.AccessCount++
	content := itm.content
	itm.mutex.Unlock()
	return content
}

// Dur returns the time since the item was last accessed.
func (itm *cacheItem) Dur() time.Duration {
	itm.mutex.RLock()
	last := itm.Lastaccess
	itm.mutex.RUnlock()
	return time.Since(last)
}

// CacheItemPair maps key to access counter.
type CacheItemPair struct {
	Key         string
	AccessCount uint64
}

type CacheItemPairList []CacheItemPair

func (p CacheItemPairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p CacheItemPairList) Len() int           { return len(p) }
func (p CacheItemPairList) Less(i, j int) bool { return p[i].AccessCount > p[j].AccessCount }
