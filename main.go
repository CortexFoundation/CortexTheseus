package main

import (
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
	storageDir := "/home/lizhen/storage"
	client := download.NewTorrentManager(storageDir)
	client.SetTrackers([]string{"http//:47.52.39.170:5008/announce"})
	monitor.InitStorage(storageDir, client)
	//	time.Sleep(time.Second * 3)
	//	client.NewTorrent <- "magnet:?xt=urn:btih:51D17FCFC86CA481AD70883083BD6BEC2ABB92AD&tr=http%3a%2f%2f47.52.39.170%3a5008%2fannounce"
	go monitor.ListenOn("http://192.168.5.11:28888", client)
	for {
		time.Sleep(time.Second * 5)
	}
	return 0
}
