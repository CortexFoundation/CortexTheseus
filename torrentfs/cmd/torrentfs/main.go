package main

import (
	"flag"
	"github.com/CortexFoundation/torrentfs"
	"os"
	"os/signal"
	"syscall"
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

	cfg := torrentfs.Config{
		DataDir:         torrentfs.DefaultConfig.DataDir,
		Host:            torrentfs.DefaultConfig.Host,
		Port:            torrentfs.DefaultConfig.Port,
		DefaultTrackers: torrentfs.DefaultConfig.DefaultTrackers,
		SyncMode:        torrentfs.DefaultConfig.SyncMode,
		TestMode:        torrentfs.DefaultConfig.TestMode,
	}

	cfg.Host = *Host
	cfg.Port = *Port
	cfg.DataDir = *Dir
	cfg.DefaultTrackers = *trackerURI

	tfs := torrentfs.New(&cfg, "")
	tfs.Start(nil)
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	for {
		<-c
		tfs.Stop()
	}
	return 0
}
