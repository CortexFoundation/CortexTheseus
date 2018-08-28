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

	storageDir := flag.String("D", "/data/serving/InferenceServer/warehouse/", "Storage path")
	rpcURI := flag.String("R", "http://localhost:28888", "json-rpc uri")
	flag.Parse()
	client := download.NewTorrentManager(*storageDir)
	client.SetTrackers([]string{"http//:47.52.39.170:5008/announce"})
	monitor.InitStorage(*storageDir, client)
	//	time.Sleep(time.Second * 3)
	//	client.NewTorrent <- "magnet:?xt=urn:btih:51D17FCFC86CA481AD70883083BD6BEC2ABB92AD&tr=http%3a%2f%2f47.52.39.170%3a5008%2fannounce"
	go monitor.ListenOn(*rpcURI, client)
	for {
		time.Sleep(time.Second * 5)
	}
	return 0
}
