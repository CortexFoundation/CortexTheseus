package filecache

import (
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"path/filepath"
)

// HttpHandler returns a valid HTTP handler for the given cache.
func HttpHandler(cache *FileCache) func(http.ResponseWriter, *http.Request) {
	return func(w http.ResponseWriter, r *http.Request) {
		cache.HttpWriteFile(w, r)
	}
}

func (cache *FileCache) HttpWriteFile(w http.ResponseWriter, r *http.Request) {
	path, err := url.QueryUnescape(r.URL.String())
	if err != nil {
		http.ServeFile(w, r, r.URL.Path)
	} else if len(path) > 1 {
		path = path[1:]
	} else {
		http.ServeFile(w, r, ".")
		return
	}

	if cache.InCache(path) {
		itm := cache.items[path]
		ctype := http.DetectContentType(itm.Access())
		mtype := mime.TypeByExtension(filepath.Ext(path))
		if mtype != "" && mtype != ctype {
			ctype = mtype
		}
		header := w.Header()
		header.Set("content-length", fmt.Sprintf("%d", itm.Size))
		header.Set("content-disposition",
			fmt.Sprintf("filename=%s", filepath.Base(path)))
		header.Set("content-type", ctype)
		w.Write(itm.Access())
		return
	}
	cache.Cache(path, nil)
	http.ServeFile(w, r, path)
}
