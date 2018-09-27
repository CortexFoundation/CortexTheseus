// Mounts a FUSE filesystem backed by torrents and magnet links.
package main

import (
	"github.com/anacrolix/missinggo/slices"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
	"log"
	"net"
	"os"
	"os/signal"
	"os/user"
	"path"
	"path/filepath"
	"syscall"

	"github.com/anacrolix/tagflag"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/fs"
	"github.com/anacrolix/torrent/util/dirwatch"
)

var (
	args = struct {
		DataDir string `help:"torrent files in this location describe the contents of download files"`

		DisableTrackers bool
		ReadaheadBytes  tagflag.Bytes
		ListenAddr      *net.TCPAddr
	}{
		DataDir: func() string {
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
		log.Println("Waiting for exit")
		<-c
		fs.Destroy()
		return
	}
}

func main() {
	os.Exit(mainExitCode())
}

func mainExitCode() int {
	tagflag.Parse(&args)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cfg := torrent.NewDefaultClientConfig()
	cfg.DataDir = args.DataDir
	cfg.DisableTrackers = args.DisableTrackers
	cfg.SetListenAddr(args.ListenAddr.String())
	cfg.Seed = true
	client, err := torrent.NewClient(cfg)
	if err != nil {
		log.Print(err)
		return 1
	}
	dw, err := dirwatch.New(args.DataDir)
	if err != nil {
		log.Printf("error watching torrent dir: %s", err)
		return 1
	}
	go func() {
		for ev := range dw.Events {
			switch ev.Change {
			case dirwatch.Added:
				if ev.TorrentFilePath != "" {
					filePath := ev.TorrentFilePath
					torrentPath := path.Join(filePath, "torrent")
					if _, err := os.Stat(torrentPath); err == nil {
						mi, err := metainfo.LoadFromFile(torrentPath)
						if err != nil {
							log.Printf("error adding torrent to client: %s", err)
							continue
						}
						spec := torrent.TorrentSpecFromMetaInfo(mi)
						ih := spec.InfoHash
						log.Println("Torrent", ih, "is seeding.")

						spec.Storage = storage.NewFile(filePath)
						t, _, err := client.AddTorrentSpec(spec)
						if err != nil {
							log.Printf("error adding torrent to client: %s", err)
							continue
						}
						var ss []string
						slices.MakeInto(&ss, mi.Nodes)
						t.DownloadAll()
					}
					if err != nil {
						log.Printf("error adding torrent to client: %s", err)
					}
				}
			case dirwatch.Removed:
				if ev.TorrentFilePath != "" {
					filePath := ev.TorrentFilePath
					torrentPath := path.Join(filePath, "torrent")
					if _, err := os.Stat(torrentPath); err == nil {
						mi, err := metainfo.LoadFromFile(torrentPath)
						if err != nil {
							log.Printf("error adding torrent to client: %s", err)
							continue
						}
						spec := torrent.TorrentSpecFromMetaInfo(mi)
						ih := spec.InfoHash
						T, ok := client.Torrent(ih)
						if !ok {
							break
						}
						T.Drop()
					}
				}
			}
		}
	}()
	fs := torrentfs.New(client)
	exitSignalHandlers(fs)
	return 0
}
