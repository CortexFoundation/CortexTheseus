package torrentfs

import (
	"log"
	"net/http"
	"strings"
	"sync"

	sjson "github.com/bitly/go-simplejson"
)

// HTTPMonitor ... Monitor for serving http services.
type HTTPMonitor struct {
	Addr string

	server *http.Server
	//mux    *http.ServeMux
	info sjson.Json
	mu   sync.Mutex
}

// NewHTTPMonitor ... Create a http monitor instance.
func NewHTTPMonitor(addr string) *HTTPMonitor {
	return &HTTPMonitor{
		Addr: addr,
	}
}

func (m *HTTPMonitor) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.jsonHandle(w, r)
}

func (m *HTTPMonitor) Initilize() {
	m.server = &http.Server{Addr: m.Addr, Handler: m}
	log.Println("Listen on ", m.Addr)
	go func() { log.Fatal(m.server.ListenAndServe()) }()
}

func (m *HTTPMonitor) Finalize() {
}

func (m *HTTPMonitor) Update(path string, value interface{}) {
	pathKeys := strings.Split(path, ".")
	m.mu.Lock()
	m.info.SetPath(pathKeys, value)
	m.mu.Unlock()
}

func (m *HTTPMonitor) jsonHandle(w http.ResponseWriter, r *http.Request) {
	m.mu.Lock()
	json, err := m.info.EncodePretty()
	m.mu.Unlock()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")
	w.Write(json)
}
