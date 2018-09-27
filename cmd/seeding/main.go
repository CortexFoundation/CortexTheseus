// Mounts a FUSE filesystem backed by torrents and magnet links.
package main

import (
	"log"
	"net"
	"os"
	"os/signal"
	"os/user"
	"path/filepath"
	"syscall"

	"github.com/anacrolix/tagflag"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/fs"
	"github.com/anacrolix/torrent/util/dirwatch"
)

var (
	args = struct {
		MetainfoDir string `help:"torrent files in this location describe the contents of the mounted filesystem"`
		DownloadDir string `help:"location to save torrent data"`

		DisableTrackers bool
		ReadaheadBytes  tagflag.Bytes
		ListenAddr      *net.TCPAddr
	}{
		MetainfoDir: func() string {
			_user, err := user.Current()
			if err != nil {
				log.Fatal(err)
			}
			return filepath.Join(_user.HomeDir, ".torrent")
		}(),
		ReadaheadBytes: 10 << 20,
		ListenAddr:     &net.TCPAddr{},
	}
)

func exitSignalHandlers(fs *torrentfs.TorrentFS) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	for {
		<-c
		fs.Destroy()
	}
}

func main() {
	os.Exit(mainExitCode())
}

func mainExitCode() int {
	tagflag.Parse(&args)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	// TODO: Think about the ramifications of exiting not due to a signal.
	cfg := torrent.NewDefaultClientConfig()
	cfg.DataDir = args.DownloadDir
	cfg.DisableTrackers = args.DisableTrackers
	cfg.SetListenAddr(args.ListenAddr.String())
	cfg.Seed = true
	client, err := torrent.NewClient(cfg)
	if err != nil {
		log.Print(err)
		return 1
	}
	dw, err := dirwatch.New(args.MetainfoDir)
	if err != nil {
		log.Printf("error watching torrent dir: %s", err)
		return 1
	}
	go func() {
		for ev := range dw.Events {
			switch ev.Change {
			case dirwatch.Added:
				if ev.TorrentFilePath != "" {
					t, err := client.AddTorrentFromFile(ev.TorrentFilePath)
					t.DownloadAll()
					if err != nil {
						log.Printf("error adding torrent to client: %s", err)
					}
				} else if ev.MagnetURI != "" {
					t, err := client.AddMagnet(ev.MagnetURI)
					t.DownloadAll()
					if err != nil {
						log.Printf("error adding magnet: %s", err)
					}
				}
			case dirwatch.Removed:
				T, ok := client.Torrent(ev.InfoHash)
				if !ok {
					break
				}
				T.Drop()
			}
		}
	}()
	fs := torrentfs.New(client)
	go exitSignalHandlers(fs)
	return 0
}
