package main

import (
	"log"
	"net"
	"os"

	"github.com/anacrolix/torrent"
)

func main() {
	os.Exit(mainExitCode())
}

func Download(t *torrent.Client, mURI string) {
	log.Println("Down")
	tm, err := t.AddMagnet(mURI)
	if err != nil {
		log.Printf("error adding magnet: %s", err)
	}

	go func() {
		<-tm.GotInfo()
		tm.DownloadAll()
	}()

}
func mainExitCode() int {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableTCP = true
	cfg.DataDir = "./"
	cfg.DisableEncryption = true
	listenAddr := &net.TCPAddr{}
	log.Println(listenAddr)
	cfg.SetListenAddr(listenAddr.String())
	t, err := torrent.NewClient(cfg)
	if err != nil {
		log.Println(err)
	}
	torrentFiles := make(chan string)
	go func() {
		for {
			torrent := <-torrentFiles
			go Download(t, torrent)
		}
	}()
	ListenOn("http://192.168.5.11:28888", torrentFiles)
	return 0
}
