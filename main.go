package main

import (
	"flag"
	"os"
	"time"

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
	flag.Parse()
	client := download.NewTorrentManager(*storageDir)
	client.SetTrackers([]string{"http//:47.52.39.170:5008/announce"})
	monitor.InitStorage(*storageDir, client)
	go monitor.ListenOn(*rpcURI, client)
	for {
		time.Sleep(time.Second * 5)
	}
	return 0
}
