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
	client := download.NewManager(torrentFiles)
	monitor.ListenOn("http://192.168.5.11:28888", client)
	return 0
}
