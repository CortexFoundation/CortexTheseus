package filecache

import (
	"errors"
	"os"
	"path"
	"path/filepath"
	"sync"
	"syscall"
	"time"
	"io/fs"

	"github.com/anacrolix/log"

	"github.com/anacrolix/missinggo/v2/pproffd"
	"github.com/anacrolix/missinggo/v2/resource"
)

const (
	dirPerm  = 0755
	filePerm = 0644
)

type Cache struct {
	root     string
	mu       sync.Mutex
	capacity int64
	filled   int64
	policy   Policy
	items    map[key]itemState
}

type CacheInfo struct {
	Capacity int64
	Filled   int64
	NumItems int
}

type ItemInfo struct {
	Path     key
	Accessed time.Time
	Size     int64
}

// Calls the function for every item known to be in the cache.
func (me *Cache) WalkItems(cb func(ItemInfo)) {
	me.mu.Lock()
	defer me.mu.Unlock()
	for k, ii := range me.items {
		cb(ItemInfo{
			Path:     k,
			Accessed: ii.Accessed,
			Size:     ii.Size,
		})
	}
}

func (me *Cache) Info() (ret CacheInfo) {
	me.mu.Lock()
	defer me.mu.Unlock()
	ret.Capacity = me.capacity
	ret.Filled = me.filled
	ret.NumItems = len(me.items)
	return
}

// Setting a negative capacity means unlimited.
func (me *Cache) SetCapacity(capacity int64) {
	me.mu.Lock()
	defer me.mu.Unlock()
	me.capacity = capacity
}

func NewCache(root string) (ret *Cache, err error) {
	root, err = filepath.Abs(root)
	ret = &Cache{
		root:     root,
		capacity: -1, // unlimited
	}
	ret.mu.Lock()
	go func() {
		defer ret.mu.Unlock()
		ret.rescan()
	}()
	return
}

// An empty return path is an error.
func sanitizePath(p string) (ret key) {
	if p == "" {
		return
	}
	ret = key(path.Clean("/" + p))
	if ret[0] == '/' {
		ret = ret[1:]
	}
	return
}

// Leaf is a descendant of root.
func pruneEmptyDirs(root string, leaf string) (err error) {
	rootInfo, err := os.Stat(root)
	if err != nil {
		return
	}
	for {
		var leafInfo os.FileInfo
		leafInfo, err = os.Stat(leaf)
		if os.IsNotExist(err) {
			goto parent
		}
		if err != nil {
			return
		}
		if !leafInfo.IsDir() {
			return
		}
		if os.SameFile(rootInfo, leafInfo) {
			return
		}
		if os.Remove(leaf) != nil {
			return
		}
	parent:
		leaf = filepath.Dir(leaf)
	}
}

func (me *Cache) Remove(path string) error {
	me.mu.Lock()
	defer me.mu.Unlock()
	return me.remove(sanitizePath(path))
}

var (
	ErrBadPath = errors.New("bad path")
	ErrIsDir   = errors.New("is directory")
)

func (me *Cache) StatFile(path string) (os.FileInfo, error) {
	return os.Stat(me.realpath(sanitizePath(path)))
}

func isMissingDir(err error) bool {
	// I'm not sure why we get EINVAL for missing path components sometimes. It happens on MacOS. I
	// wonder if it would happen the same on other OS.
	return errors.Is(err, syscall.EINVAL) || errors.Is(err, syscall.ENOENT) || errors.Is(err, fs.ErrNotExist)
}

func (me *Cache) OpenFile(path string, flag int) (ret *File, err error) {
	key := sanitizePath(path)
	if key == "" {
		err = ErrIsDir
		return
	}
	filePath := me.realpath(key)
	f, err := os.OpenFile(filePath, flag, filePerm)
	// Ensure intermediate directories and try again.
	if flag&os.O_CREATE != 0 && isMissingDir(err) {
		dirPath := filepath.Dir(filePath)
		if dirPath == "" {
			panic(key)
		}
		defer func() {
			if err != nil {
				go me.pruneEmptyDirs(key)
			}
		}()
		for {
			err = os.MkdirAll(dirPath, dirPerm)
			if isMissingDir(err) {
				continue
			} else if err != nil {
				return
			}
			f, err = os.OpenFile(filePath, flag, filePerm)
			if isMissingDir(err) {
				continue
			}
			break
		}
	}
	if err != nil {
		return
	}
	ret = &File{
		path: key,
		f:    pproffd.WrapOSFile(f),
		onRead: func(n int) {
			me.mu.Lock()
			defer me.mu.Unlock()
			me.updateItem(key, func(i *itemState, ok bool) bool {
				i.Accessed = time.Now()
				return ok
			})
		},
		afterWrite: func(endOff int64) {
			me.mu.Lock()
			defer me.mu.Unlock()
			me.updateItem(key, func(i *itemState, ok bool) bool {
				i.Accessed = time.Now()
				if endOff > i.Size {
					i.Size = endOff
				}
				return ok
			})
		},
	}
	me.mu.Lock()
	defer me.mu.Unlock()
	me.updateItem(key, func(i *itemState, ok bool) bool {
		if !ok {
			*i, ok = me.statKey(key)
		}
		i.Accessed = time.Now()
		return ok
	})
	return
}

func (me *Cache) rescan() {
	me.filled = 0
	me.policy = new(lru)
	me.items = make(map[key]itemState)
	err := filepath.Walk(me.root, func(path string, info os.FileInfo, err error) error {
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		path, err = filepath.Rel(me.root, path)
		if err != nil {
			log.Print(err)
			return nil
		}
		key := sanitizePath(path)
		me.updateItem(key, func(i *itemState, ok bool) bool {
			if ok {
				panic("scanned duplicate items")
			}
			*i, ok = me.statKey(key)
			return ok
		})
		return nil
	})
	if err != nil {
		panic(err)
	}
}

func (me *Cache) statKey(k key) (i itemState, ok bool) {
	fi, err := os.Stat(me.realpath(k))
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		panic(err)
	}
	i.FromOSFileInfo(fi)
	ok = true
	return
}

func (me *Cache) updateItem(k key, u func(*itemState, bool) bool) {
	ii, ok := me.items[k]
	me.filled -= ii.Size
	if u(&ii, ok) {
		me.filled += ii.Size
		me.policy.Used(k, ii.Accessed)
		me.items[k] = ii
	} else {
		me.policy.Forget(k)
		delete(me.items, k)
	}
	me.trimToCapacity()
}

func (me *Cache) realpath(path key) string {
	return filepath.Join(me.root, filepath.FromSlash(string(path)))
}

func (me *Cache) TrimToCapacity() {
	me.mu.Lock()
	defer me.mu.Unlock()
	me.trimToCapacity()
}

func (me *Cache) pruneEmptyDirs(path key) {
	pruneEmptyDirs(me.root, me.realpath(path))
}

func (me *Cache) remove(path key) error {
	err := os.Remove(me.realpath(path))
	if os.IsNotExist(err) {
		err = nil
	}
	if err != nil {
		return err
	}
	me.pruneEmptyDirs(path)
	me.updateItem(path, func(*itemState, bool) bool {
		return false
	})
	return nil
}

func (me *Cache) trimToCapacity() {
	if me.capacity < 0 {
		return
	}
	for me.filled > me.capacity {
		me.remove(me.policy.Choose().(key))
	}
}

// TODO: Do I need this?
func (me *Cache) pathInfo(p string) itemState {
	return me.items[sanitizePath(p)]
}

func (me *Cache) Rename(from, to string) (err error) {
	_from := sanitizePath(from)
	_to := sanitizePath(to)
	me.mu.Lock()
	defer me.mu.Unlock()
	err = os.MkdirAll(filepath.Dir(me.realpath(_to)), dirPerm)
	if err != nil {
		return
	}
	err = os.Rename(me.realpath(_from), me.realpath(_to))
	if err != nil {
		return
	}
	// We can do a dance here to copy the state from the old item, but lets
	// just stat the new item for now.
	me.updateItem(_from, func(i *itemState, ok bool) bool {
		return false
	})
	me.updateItem(_to, func(i *itemState, ok bool) bool {
		*i, ok = me.statKey(_to)
		return ok
	})
	return
}

func (me *Cache) Stat(path string) (os.FileInfo, error) {
	return os.Stat(me.realpath(sanitizePath(path)))
}

func (me *Cache) AsResourceProvider() resource.Provider {
	return &uniformResourceProvider{me}
}
