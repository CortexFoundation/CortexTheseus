package monitor

import (
	sjson "github.com/bitly/go-simplejson"
	"log"
	"net/http"
	"strings"
	"sync"
)

type HttpMonitor struct {
	Addr string

	server *http.Server
	mux    *http.ServeMux
	info   sjson.Json
	mtx    sync.Mutex
}

func NewHttpMonitor(addr string) *HttpMonitor {
	return &HttpMonitor{
		Addr: addr,
	}
}

func (m *HttpMonitor) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	m.jsonHandle(w, r)
}

func (m *HttpMonitor) Initilize() {
	m.server = &http.Server{Addr: m.Addr, Handler: m}
	log.Println("Listen on ", m.Addr)
	go func() { log.Fatal(m.server.ListenAndServe()) }()
}

func (m *HttpMonitor) Finalize() {
}

func (m *HttpMonitor) Update(path string, value interface{}) {
	pathKeys := strings.Split(path, ".")
	m.mtx.Lock()
	m.info.SetPath(pathKeys, value)
	m.mtx.Unlock()
}

func (m *HttpMonitor) jsonHandle(w http.ResponseWriter, r *http.Request) {
	m.mtx.Lock()
	json, err := m.info.EncodePretty()
	m.mtx.Unlock()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Content-Type", "application/json")
	w.Write(json)
}
