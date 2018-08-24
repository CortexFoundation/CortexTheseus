package main

import (
	"os"

	download "./manager"
	monitor "./monitor"
)

func main() {
	os.Exit(mainExitCode())
}

func mainExitCode() int {
	torrentFiles := make(chan string)
	storageDir := "/home/lizhen/storage"
	client := download.NewManager(storageDir, torrentFiles)
	monitor.InitStorage(storageDir, client)
	monitor.ListenOn("http://192.168.5.11:28888", client)
	return 0
}
