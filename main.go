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
	t := download.Client()
	torrentFiles := make(chan string)
	go func() {
		for {
			torrent := <-torrentFiles
			go download.Download(t, torrent)
		}
	}()
	monitor.ListenOn("http://192.168.5.11:28888", torrentFiles)
	return 0
}
