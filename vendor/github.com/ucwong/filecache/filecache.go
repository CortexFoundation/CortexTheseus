package filecache

import (
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"time"
)

// FileCache represents a cache in memory.
// An ExpireItem value of 0 means that items should not be expired based
// on time in memory.
type FileCache struct {
	dur        time.Duration
	items      map[string]*cacheItem
	in         chan *CacheInfo
	mutex      sync.RWMutex
	shutdown   chan struct{}
	wg         sync.WaitGroup
	MaxItems   int   // Maximum number of files to cache
	MaxSize    int64 // Maximum file size to store
	ExpireItem int   // Seconds a file should be cached for
	Every      int   // Run an expiration check Every seconds
}

type CacheInfo struct {
	name    string
	content []byte
}

// NewDefaultCache returns a new FileCache with sane defaults.
func NewDefaultCache() *FileCache {
	return &FileCache{
		dur:        0,
		items:      nil,
		in:         nil,
		MaxItems:   DefaultMaxItems,
		MaxSize:    DefaultMaxSize,
		ExpireItem: DefaultExpireItem,
		Every:      DefaultEvery,
	}
}

func (cache *FileCache) isCacheNull() bool {
	cache.mutex.RLock()
	defer cache.mutex.RUnlock()
	return cache.items == nil
}

func (cache *FileCache) getItem(name string) (itm *cacheItem, ok bool) {
	cache.mutex.RLock()
	defer cache.mutex.RUnlock()
	if cache.items == nil {
		return nil, false
	}
	itm, ok = cache.items[name]
	return
}

// addItem is an internal function for adding an item to the cache.
// IMPORTANT: do not hold cache mutex while calling functions that may lock again.
func (cache *FileCache) addItem(name string, content []byte) (err error) {
	// if cache not started or destroyed, no-op
	if cache.isCacheNull() {
		return
	}

	// If already in cache and not expired, nothing to do.
	if cache.InCache(name) && !cache.itemExpired(name) {
		return
	}
	// If in cache but expired, ensure it is removed before adding.
	if cache.InCache(name) {
		// deleteItem acquires write lock internally
		cache.deleteItem(name)
	}

	// Create cache item (this may be expensive; do it outside locks)
	itm, err := cacheFile(name, cache.MaxSize, content)
	if err != nil || itm == nil {
		return err
	}

	// Insert under lock
	cache.mutex.Lock()
	// double-check items map exists (in case Stop() ran concurrently)
	if cache.items != nil {
		cache.items[name] = itm
	}
	cache.mutex.Unlock()

	return nil
}

func (cache *FileCache) deleteItem(name string) {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()
	if cache.items == nil {
		return
	}
	delete(cache.items, name)
}

// itemListener is a goroutine that listens for incoming files and caches
// them.
func (cache *FileCache) itemListener() {
	for {
		select {
		case c := <-cache.in:
			// addItem is safe to call concurrently; it manages its own locking.
			_ = cache.addItem(c.name, c.content)
		case <-cache.shutdown:
			return
		}
	}
}

// expireOldest is used to expire the oldest item in the cache.
// The force argument is used to indicate it should remove at least one
// entry; for example, if a large number of files are cached at once, none
// may appear older than another.
func (cache *FileCache) expireOldest(force bool) {
	cache.mutex.Lock()
	defer cache.mutex.Unlock()

	if cache.items == nil || len(cache.items) == 0 {
		return
	}

	var (
		oldestName string
		oldest     time.Time
		first      = true
	)

	for name, itm := range cache.items {
		if first || itm.Lastaccess.Before(oldest) {
			oldest = itm.Lastaccess
			oldestName = name
			first = false
		}
		// if force requested and we found any candidate, we still continue to find the true oldest
	}
	if oldestName != "" {
		delete(cache.items, oldestName)
	}
}

// vacuum is a background goroutine responsible for cleaning the cache.
// It runs periodically, every cache.Every seconds. If cache.Every is set
// to 0, it will not run.
func (cache *FileCache) vacuum() {
	if cache.Every < 1 {
		return
	}

	ticker := time.NewTicker(cache.dur)
	defer ticker.Stop()

	for {
		select {
		case <-cache.shutdown:
			return
		case <-ticker.C:
			// If cache destroyed, exit.
			if cache.isCacheNull() {
				return
			}

			// Collect expired names under read lock to minimize write lock time.
			var expired []string
			cache.mutex.RLock()
			for name := range cache.items {
				// we call itemExpired which may call os.Stat and getItem (safe)
				// so do that outside of holding the read lock for long time:
				expired = append(expired, name)
			}
			cache.mutex.RUnlock()

			// Evaluate expirations and delete under write lock as needed.
			for _, name := range expired {
				if cache.itemExpired(name) {
					cache.deleteItem(name)
				}
			}

			// Ensure max items constraint is enforced.
			for cache.Size() > cache.MaxItems {
				cache.expireOldest(true)
			}
		}
	}
}

// FileChanged returns true if file should be expired based on mtime.
// If the file has changed on disk or no longer exists, it should be
// expired.
func (cache *FileCache) changed(name string) bool {
	itm, ok := cache.getItem(name)
	if !ok || itm == nil {
		return true
	}
	fi, err := os.Stat(name)
	if err != nil {
		return true
	}
	// WasModified should be safe to call on itm.
	if !itm.WasModified(fi) {
		return true
	}
	return false
}

// Expired returns true if the item has not been accessed recently.
func (cache *FileCache) expired(name string) bool {
	itm, ok := cache.getItem(name)
	if !ok || itm == nil {
		return true
	}
	sec := int(itm.Dur().Seconds())
	if cache.ExpireItem == 0 {
		return false
	}
	if sec >= cache.ExpireItem {
		return true
	}
	return false
}

// itemExpired returns true if an item is expired.
func (cache *FileCache) itemExpired(name string) bool {
	// if changed or (ExpireItem configured and expired by time)
	if cache.changed(name) {
		return true
	}
	if cache.ExpireItem != 0 && cache.expired(name) {
		return true
	}
	return false
}

// Active returns true if the cache has been started, and false otherwise.
func (cache *FileCache) Active() bool {
	if cache.in == nil || cache.isCacheNull() {
		return false
	}
	return true
}

// Size returns the number of entries in the cache.
func (cache *FileCache) Size() int {
	cache.mutex.RLock()
	defer cache.mutex.RUnlock()
	if cache.items == nil {
		return 0
	}
	return len(cache.items)
}

// FileSize returns the sum of the file sizes stored in the cache
func (cache *FileCache) FileSize() (totalSize int64) {
	cache.mutex.RLock()
	defer cache.mutex.RUnlock()
	if cache.items == nil {
		return 0
	}
	for _, itm := range cache.items {
		totalSize += itm.Size
	}
	return
}

// StoredFiles returns the list of files stored in the cache.
func (cache *FileCache) StoredFiles() (fileList []string) {
	cache.mutex.RLock()
	if cache.items == nil {
		cache.mutex.RUnlock()
		return nil
	}
	fileList = make([]string, 0, len(cache.items))
	for name := range cache.items {
		fileList = append(fileList, name)
	}
	cache.mutex.RUnlock()
	return
}

// InCache returns true if the item is in the cache.
func (cache *FileCache) InCache(name string) bool {
	// changed() reads item info via getItem and os.Stat; safe to call without holding write lock.
	if cache.changed(name) {
		// delete if changed
		cache.deleteItem(name)
		return false
	}
	cache.mutex.RLock()
	defer cache.mutex.RUnlock()
	if cache.items == nil {
		return false
	}
	_, ok := cache.items[name]
	return ok
}

// WriteItem writes the cache item to the specified io.Writer.
func (cache *FileCache) WriteItem(w io.Writer, name string) (err error) {
	itm, ok := cache.getItem(name)
	if !ok || itm == nil {
		if !SquelchItemNotInCache {
			err = ItemNotInCache
		}
		return
	}

	r := itm.GetReader()
	itm.mutex.Lock()
	itm.Lastaccess = time.Now()
	itm.mutex.Unlock()

	// io.Copy can be slow; don't hold itm lock during I/O.
	n, err := io.Copy(w, r)
	if err != nil {
		return err
	} else if int64(n) != itm.Size {
		return WriteIncomplete
	}
	return nil
}

// GetItem returns the content of the item and a bool if name is present.
// GetItem should be used when you are certain an object is in the cache,
// or if you want to use the cache only.
func (cache *FileCache) GetItem(name string) (content []byte, ok bool) {
	itm, ok := cache.getItem(name)
	if !ok || itm == nil {
		return nil, false
	}
	content = itm.Access()
	return
}

// GetItemString is the same as GetItem, except returning a string.
func (cache *FileCache) GetItemString(name string) (content string, ok bool) {
	itm, ok := cache.getItem(name)
	if !ok || itm == nil {
		return "", false
	}
	content = string(itm.Access())
	return
}

// ReadFileString is the same as ReadFile, except returning a string.
func (cache *FileCache) ReadFileString(name string) (content string, err error) {
	raw, err := cache.ReadFile(name)
	if err == nil {
		content = string(raw)
	}
	return
}

// WriteFile writes the file named by 'name' to the specified io.Writer.
// If the file is in the cache, it is loaded from the cache; otherwise,
// it is read from the filesystem and the file is cached in the background.
func (cache *FileCache) WriteFile(w io.Writer, name string) (err error) {
	if cache.InCache(name) {
		return cache.WriteItem(w, name)
	}

	var fi os.FileInfo
	fi, err = os.Stat(name)
	if err != nil {
		return
	} else if fi.IsDir() {
		return ItemIsDirectory
	}
	// async cache
	cache.Cache(name, nil)
	var file *os.File
	file, err = os.Open(name)
	if err != nil {
		return
	}
	defer file.Close()
	_, err = io.Copy(w, file)
	return
}

// Cache will store the file named by 'name' to the cache.
// This function doesn't return anything as it passes the file onto the
// incoming pipe; the file will be cached asynchronously. Errors will
// not be returned.
func (cache *FileCache) Cache(name string, content []byte) {
	// Maintain max items; expire oldest if necessary.
	if cache.Size() >= cache.MaxItems {
		cache.expireOldest(true)
	}
	// It may block if channel is full, but that's acceptable design here.
	cache.in <- &CacheInfo{name, content}
}

// CacheNow immediately caches the file named by 'name'.
func (cache *FileCache) CacheNow(name string) (err error) {
	if cache.Size() >= cache.MaxItems {
		cache.expireOldest(true)
	}
	return cache.addItem(name, nil)
}

// Start activates the file cache; it will start up the background caching
// and automatic cache expiration goroutines and initialise the internal
// data structures.
func (cache *FileCache) Start() error {
	dur, err := time.ParseDuration(fmt.Sprintf("%ds", cache.Every))
	if err != nil {
		return err
	}
	cache.dur = dur
	cache.items = make(map[string]*cacheItem, 0)
	cache.in = make(chan *CacheInfo, NewCachePipeSize)
	cache.shutdown = make(chan struct{})

	cache.wg.Add(2)
	go func() {
		defer cache.wg.Done()
		cache.itemListener()
	}()
	go func() {
		defer cache.wg.Done()
		cache.vacuum()
	}()
	return nil
}

// Stop turns off the file cache.
// This closes the concurrent caching mechanism, destroys the cache, and
// the background scanner that it should stop.
// If there are any items or cache operations ongoing while Stop() is called,
// it is undefined how they will behave.
func (cache *FileCache) Stop() {
	// Closing shutdown notifies goroutines to stop.
	close(cache.shutdown)

	// Wait for background goroutines.
	cache.wg.Wait()

	// Clear items safely.
	cache.mutex.Lock()
	if cache.items != nil {
		for name := range cache.items {
			delete(cache.items, name)
		}
		cache.items = nil
	}
	cache.mutex.Unlock()
}

// RemoveItem immediately removes the item from the cache if it is present.
// It returns a boolean indicating whether anything was removed, and an error
// if an error has occurred.
func (cache *FileCache) Remove(name string) (ok bool, err error) {
	cache.mutex.RLock()
	if cache.items == nil {
		cache.mutex.RUnlock()
		return false, nil
	}
	_, ok = cache.items[name]
	cache.mutex.RUnlock()
	if !ok {
		return false, nil
	}
	cache.deleteItem(name)
	_, still := cache.getItem(name)
	return !still, nil
}

// MostAccessed returns a slice of the "count" most frequently accessed cache items.
// The cache mutex is locked while this method executes.
func (cache *FileCache) MostAccessed(count int64) []*cacheItem {
	cache.mutex.RLock()
	defer cache.mutex.RUnlock()

	if cache.items == nil || count <= 0 {
		return nil
	}

	type pair struct {
		key   string
		count uint64
	}
	p := make([]pair, 0, len(cache.items))
	for k, v := range cache.items {
		p = append(p, pair{k, v.AccessCount})
	}
	// sort descending by access count
	sort.Slice(p, func(i, j int) bool {
		return p[i].count > p[j].count
	})

	limit := int(count)
	if limit > len(p) {
		limit = len(p)
	}
	result := make([]*cacheItem, 0, limit)
	for i := 0; i < limit; i++ {
		if item, ok := cache.items[p[i].key]; ok && item != nil {
			result = append(result, item)
		}
	}
	return result
}

func (cache *FileCache) vacuumOnce() {
	if cache.isCacheNull() {
		return
	}
	for name := range cache.items {
		if cache.itemExpired(name) {
			cache.deleteItem(name)
		}
	}
	for cache.Size() > cache.MaxItems {
		cache.expireOldest(true)
	}
}
