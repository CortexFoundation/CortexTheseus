//go:build go1.18

package filecache

import (
	"os"
	"time"
)

func cacheFile(path string, maxSize int64, c []byte) (itm *cacheItem, err error) {
	fi, err := os.Stat(path)
	if err != nil {
		return
	} else if fi.Mode().IsDir() {
		err = ItemIsDirectory
		return
	} else if fi.Size() > maxSize {
		err = ItemTooLarge
		return
	}

	if len(c) > 0 {
		itm = &cacheItem{
			name:       path,
			content:    c,
			Size:       fi.Size(),
			ModTime:    fi.ModTime(),
			Lastaccess: time.Now(),
		}
		return
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return
	}

	itm = &cacheItem{
		name:       path,
		content:    content,
		Size:       fi.Size(),
		ModTime:    fi.ModTime(),
		Lastaccess: time.Now(),
	}
	return
}

// ReadFile retrieves the file named by 'name'.
// If the file is not in the cache, load the file and cache the file in the
// background. If the file was not in the cache and the read was successful,
// the error ItemNotInCache is returned to indicate that the item was pulled
// from the filesystem and not the cache, unless the SquelchItemNotInCache
// global option is set; in that case, returns no error.
func (cache *FileCache) ReadFile(name string) (content []byte, err error) {
	if cache.InCache(name) {
		content, _ = cache.GetItem(name)
	} else {
		content, err = os.ReadFile(name)
		if err == nil {
			if !SquelchItemNotInCache {
				err = ItemNotInCache
			}

			// async
			go func(n string, c []byte) {
				cache.Cache(n, c)
			}(name, content)
		} else {
			// TODO whether try to cache file async ?
		}
	}
	return
}
