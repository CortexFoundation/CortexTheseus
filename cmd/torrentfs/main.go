package main

import (
	"flag"
	"github.com/ethereum/go-ethereum/log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/CortexFoundation/torrentfs"
)

func main() {
	os.Exit(mainExitCode())
}

func mainExitCode() int {
	// DataDir := "/data/serving/InferenceServer/warehouse"
	Host := flag.String("h", "localhost", "host")
	Port := flag.Int("p", 8085, "port")
	Dir := flag.String("d", "/data", "data dir")
	trackerURI := flag.String("t", "http://47.52.39.170:5008/announce", "tracker uri")
	flag.Parse()

	trackers := strings.Split(*trackerURI, ",")
	cfg = torrentfs.DefaultConfig
	cfg.Host = *Host
	cfg.Port = *Port
	cfg.DataDir = *Dir

	tfs := torrentfs.New(&cfg, "")
	tfs.Start(nil)
	for {
	}
	return 0
}
