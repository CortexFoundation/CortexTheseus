package main

import (
	"flag"
	"os"

	download "./manager"
	monitor "./monitor"
)

func main() {
	os.Exit(mainExitCode())
}

func mainExitCode() int {
	// storageDir := "/data/serving/InferenceServer/warehouse"
	storageDir := flag.String("d", "/home/lizhen/storage", "storage path")
	rpcURI := flag.String("r", "http://192.168.5.11:28888", "json-rpc uri")
	trackerURI := flag.String("t", "http://47.52.39.170:5008/announce", "tracker uri")
	flag.Parse()
	dlCilent := download.NewTorrentManager(*storageDir)
	dlCilent.SetTrackers([]string{*trackerURI})
	monitor.InitStorage(*storageDir, dlCilent)
	m := monitor.NewMonitor()
	m.SetRPCServer(rpcURI)
	m.SetDownloader(dlCilent)
	m.Start()
	return 0
}
