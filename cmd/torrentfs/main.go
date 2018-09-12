package main

import (
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/p2p"
	"github.com/ethereum/go-ethereum/rpc"
	"github.com/CortexFoundation/CortexTheseus/torrentfs"
)

func main() {
	os.Exit(mainExitCode())
}

func mainExitCode() int {
	// DataDir := "/data/serving/InferenceServer/warehouse"
	DataDir := flag.String("d", "/home/lizhen/storage", "storage path")
	RpcURI := flag.String("r", "http://192.168.5.11:28888", "json-rpc uri")
	IpcPath := flag.String("i", "", "ipc socket path")
	trackerURI := flag.String("t", "http://47.52.39.170:5008/announce", "tracker uri")
	flag.Parse()

	trackers := strings.Split(*trackerURI, ",")
	f := &torrentfs.Flag{
		DataDir,
		RpcURI,
		IpcPath,
		&trackers,
	}

	dlCilent := torrentfs.NewTorrentManager(f)
	m := torrentfs.NewMonitor(f)
	m.SetDownloader(dlCilent)
	m.Start()
	return 0
}

var nextID uint32 // Next connection id

// Dashboard contains the dashboard internals.
type TorrentFS struct {
	lock     sync.RWMutex // Lock protecting the dashboard's internals

	logdir string

	quit chan chan error // Channel used for graceful exit
	wg   sync.WaitGroup
}

// New creates a new dashboard instance with the given configuration.
func New(commit string, logdir string) *TorrentFS {
	return &TorrentFS{
		quit:   make(chan chan error),
		logdir: logdir,
	}
}

// Protocols implements the node.Service interface.
func (db *TorrentFS) Protocols() []p2p.Protocol { return nil }

// APIs implements the node.Service interface.
func (db *TorrentFS) APIs() []rpc.API { return nil }

// Start starts the data collection thread and the listening server of the dashboard.
// Implements the node.Service interface.
func (db *TorrentFS) Start(server *p2p.Server) error {
	log.Info("Starting dashboard")

	db.wg.Add(2)
	http.HandleFunc("/", db.webHandler)

	return nil
}

// Stop stops the data collection thread and the connection listener of the dashboard.
// Implements the node.Service interface.
func (db *TorrentFS) Stop() error {
	// Close the connection listener.
	var errs []error
	// Close the collectors.
	errc := make(chan error, 1)
	for i := 0; i < 2; i++ {
		db.quit <- errc
		if err := <-errc; err != nil {
			errs = append(errs, err)
		}
	}

	// Wait until every goroutine terminates.
	db.wg.Wait()
	log.Info("Dashboard stopped")

	var err error
	if len(errs) > 0 {
		err = fmt.Errorf("%v", errs)
	}

	return err
}

// webHandler handles all non-api requests, simply flattening and returning the dashboard website.
func (db *TorrentFS) webHandler(w http.ResponseWriter, r *http.Request) {
	log.Debug("Request", "URL", r.URL)

	path := r.URL.String()
	if path == "/" {
		path = "/index.html"
	}
}
