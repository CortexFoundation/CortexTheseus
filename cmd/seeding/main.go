// Mounts a FUSE filesystem backed by torrents and magnet links.
package main

import (
	xlog "github.com/anacrolix/log"
	"github.com/anacrolix/missinggo/v2"
	"github.com/fsnotify/fsnotify"
	"log"
	"net"
	"os"
	"os/signal"
	"os/user"
	"path"
	"path/filepath"
	"syscall"
	"time"

	//"github.com/anacrolix/missinggo/v2/slices"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"

	"github.com/anacrolix/tagflag"

	"github.com/CortexFoundation/torrentfs/params"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/fs"
)

var (
	args = struct {
		DataDir string `help:"torrent files in this location describe the contents of download files"`

		ReadaheadBytes tagflag.Bytes
		ListenAddr     *net.TCPAddr
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

type Change uint

const (
	Added Change = iota
	Removed
)

type Event struct {
	Change
	InfoHash metainfo.Hash
	FilePath string
}

type entity struct {
	metainfo.Hash
	FilePath string
}

type Instance struct {
	w        *fsnotify.Watcher
	dirName  string
	Events   chan Event
	dirState map[metainfo.Hash]entity
}

func (i *Instance) Close() {
	i.w.Close()
}

func (i *Instance) handleEvents() {
	defer close(i.Events)
	for e := range i.w.Events {
		if e.Op == fsnotify.Create || e.Op == fsnotify.Remove {
			//			log.Printf("event: %s", e)
			go func() {
				time.Sleep(time.Second * 1)
				i.refresh()
			}()
		}
	}
}

func (i *Instance) handleErrors() {
	for err := range i.w.Errors {
		log.Printf("error in torrent directory watcher: %s", err)
	}
}

func isInfoHash(name string) bool {
	if len(name) != 40 {
		return false
	}
	return true
}

func torrentFileInfoHash(fileName string) (ih metainfo.Hash, ok bool) {
	mi, _ := metainfo.LoadFromFile(fileName)
	if mi == nil {
		return
	}
	ih = mi.HashInfoBytes()
	ok = true
	return
}

func scanDir(dirName string) (ee map[metainfo.Hash]entity) {
	d, err := os.Open(dirName)
	if err != nil {
		log.Print(err)
		return
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		log.Print(err)
		return
	}
	ee = make(map[metainfo.Hash]entity, len(names))
	addEntity := func(e entity) {
		ee[e.Hash] = e
	}

	for _, n := range names {
		fullName := filepath.Join(dirName, n)
		if isInfoHash(n) {
			torrentName := path.Join(fullName, "torrent")
			ih, ok := torrentFileInfoHash(torrentName)
			if !ok {
				continue
			}
			e := entity{
				FilePath: fullName,
			}
			missinggo.CopyExact(&e.Hash, ih)
			addEntity(e)
		}
	}
	return
}

func (i *Instance) torrentRemoved(ih metainfo.Hash) {
	i.Events <- Event{
		InfoHash: ih,
		Change:   Removed,
	}
}

func (i *Instance) torrentAdded(e entity) {
	i.Events <- Event{
		InfoHash: e.Hash,
		FilePath: e.FilePath,
		Change:   Added,
	}
}

func (i *Instance) refresh() {
	_new := scanDir(i.dirName)
	old := i.dirState
	for ih := range old {
		_, ok := _new[ih]
		if !ok {
			i.torrentRemoved(ih)
		}
	}
	for ih, newE := range _new {
		oldE, ok := old[ih]
		if ok {
			if newE == oldE {
				continue
			}
			i.torrentRemoved(ih)
		}
		i.torrentAdded(newE)
	}
	i.dirState = _new
}

func NewDirWatch(dirName string) (i *Instance, err error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return
	}
	err = w.Add(dirName)
	if err != nil {
		w.Close()
		return
	}
	i = &Instance{
		w:        w,
		dirName:  dirName,
		Events:   make(chan Event),
		dirState: make(map[metainfo.Hash]entity, 0),
	}
	go func() {
		i.refresh()
		go i.handleEvents()
		go i.handleErrors()
	}()
	return
}

func exitSignalHandlers(fs *torrentfs.TorrentFS) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	for {
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
	cfg.SetListenAddr(args.ListenAddr.String())
	cfg.Logger = xlog.Discard
	cfg.Seed = true
	cfg.DropDuplicatePeerIds = true
	//cfg.EstablishedConnsPerTorrent = 10
	//cfg.HalfOpenConnsPerTorrent = 10
	cfg.DisableUTP = true
	cfg.DisableTCP = false
	cfg.NoDHT = false
	client, err := torrent.NewClient(cfg)
	if err != nil {
		log.Print(err)
		return 1
	}
	dw, err := NewDirWatch(args.DataDir)
	if err != nil {
		log.Printf("error watching torrent dir: %s", err)
		return 1
	}

	array := make([][]string, len(params.MainnetTrackers))
	for i, tracker := range params.MainnetTrackers {
		array[i] = []string{"udp" + tracker}
		//array[i] = []string{tracker}
	}

	log.Println(array)

	go func() {
		/*
			entities := scanDir(args.DataDir)
			for _, x := range entities {
				log.Print(x.String())
			}
		*/
		for ev := range dw.Events {
			switch ev.Change {
			case Added:
				if ev.FilePath != "" {
					filePath := ev.FilePath
					torrentPath := path.Join(filePath, "torrent")
					if _, err := os.Stat(torrentPath); err == nil {
						mi, err := metainfo.LoadFromFile(torrentPath)
						if err != nil {
							log.Printf("error adding torrent to client: %s", err)
							continue
						}
						spec := torrent.TorrentSpecFromMetaInfo(mi)
						ih := spec.InfoHash
						//spec.Trackers = append(spec.Trackers, params.MainnetTrackers)
						spec.Trackers = array

						spec.Storage = storage.NewFile(filePath)
						t, _, err := client.AddTorrentSpec(spec)
						if err != nil {
							log.Printf("error adding torrent to client: %s", err)
							continue
						}
						//<-t.GotInfo()
						//t.VerifyData()
						//var ss []string
						//slices.MakeInto(&ss, mi.Nodes)
						t.DownloadAll()
						//client.WaitAll()
						//go func() {
						//	time.Sleep(time.Second * 5)
						if t.Seeding() {
							log.Println(ih, "is seeding")
						}
						//}()
					}
					if err != nil {
						log.Printf("error adding torrent to client: %s", err)
					}
				}
			case Removed:
				if t, ok := client.Torrent(ev.InfoHash); ok {
					t.Drop()
					log.Printf("Torrent %s has been removed", ev.InfoHash.String())
				}
			}
		}
	}()
	fs := torrentfs.New(client)
	exitSignalHandlers(fs)
	return 0
}
