package torrentfs

import (
	"bytes"
	"crypto/sha1"
	"fmt"
	"github.com/anacrolix/missinggo/slices"
	"github.com/bradfitz/iter"
	"github.com/edsrzf/mmap-go"
	"io"
	"net"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/mmap_span"
	"github.com/anacrolix/torrent/storage"
	"github.com/ethereum/go-ethereum/log"
)

const (
	defaultBytesLimitation          = 512 * 1024
	queryTimeInterval               = 5
	removeTorrentChanBuffer         = 16
	newTorrentChanBuffer            = 32
	updateTorrentChanBuffer         = 32
	expansionFactor         float64 = 1.5
	// Pending for gotInfo
	torrentPending     = 0
	torrentPaused      = 1
	torrentRunning     = 2
	torrentSeeding     = 3
	defaultTmpFilePath = ".tmp"
)

// Torrent ...
type Torrent struct {
	*torrent.Torrent
	bytesRequested  int64
	bytesLimitation int64
	bytesCompleted  int64
	bytesMissing    int64
	status          int64
}

func (t *Torrent) Seed() {
	t.VerifyData()
	t.DownloadAll()
	t.status = torrentSeeding
}

func (t *Torrent) Seeding() bool {
	return t.status == torrentSeeding
}

// Pause ...
func (t *Torrent) Pause() {
	if t.status != torrentPaused {
		t.status = torrentPaused
		t.Drop()
	}
}

// Paused ...
func (t *Torrent) Paused() bool {
	return t.status == torrentPaused
}

// Run ...
func (t *Torrent) Run() {
	if t.status != torrentRunning {
		t.DownloadAll()
		t.status = torrentRunning
	}
}

// Running ...
func (t *Torrent) Running() bool {
	return t.status == torrentRunning
}

// Pending ...
func (t *Torrent) Pending() bool {
	return t.status == torrentPending
}

// TorrentManager ...
type TorrentManager struct {
	client        *torrent.Client
	torrents      map[metainfo.Hash]*Torrent
	trackers      []string
	DataDir       string
	TmpDataDir    string
	closeAll      chan struct{}
	newTorrent    chan string
	removeTorrent chan string
	updateTorrent chan interface{}
	halt          bool
	mu            sync.Mutex
}

func (tm *TorrentManager) Close() error {
	close(tm.closeAll)
	log.Info("Torrent Download Manager Closed")
	return nil
}

func (tm *TorrentManager) NewTorrent(input string) error {
	tm.newTorrent <- input
	return nil
}

func (tm *TorrentManager) RemoveTorrent(input string) error {
	tm.removeTorrent <- input
	return nil
}

func (tm *TorrentManager) UpdateTorrent(input interface{}) error {
	tm.updateTorrent <- input
	return nil
}

func isMagnetURI(uri string) bool {
	return strings.HasPrefix(uri, "magnet:?xt=urn:btih:")
}

// SetTrackers ...
func (tm *TorrentManager) SetTrackers(trackers []string) {
	for _, tracker := range trackers {
		tm.trackers = append(tm.trackers, tracker)
	}
}

func mmapFile(name string) (mm mmap.MMap, err error) {
	f, err := os.Open(name)
	if err != nil {
		return
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return
	}
	if fi.Size() == 0 {
		return
	}
	return mmap.MapRegion(f, -1, mmap.RDONLY, mmap.COPY, 0)
}

func verifyTorrent(info *metainfo.Info, root string) error {
	span := new(mmap_span.MMapSpan)
	for _, file := range info.UpvertedFiles() {
		filename := filepath.Join(append([]string{root, info.Name}, file.Path...)...)
		mm, err := mmapFile(filename)
		if err != nil {
			return err
		}
		if int64(len(mm)) != file.Length {
			return fmt.Errorf("file %q has wrong length, %d / %d", filename, int64(len(mm)), file.Length)
		}
		span.Append(mm)
	}
	for i := range iter.N(info.NumPieces()) {
		p := info.Piece(i)
		hash := sha1.New()
		_, err := io.Copy(hash, io.NewSectionReader(span, p.Offset(), p.Length()))
		if err != nil {
			return err
		}
		good := bytes.Equal(hash.Sum(nil), p.Hash().Bytes())
		if !good {
			return fmt.Errorf("hash mismatch at piece %d", i)
		}
	}
	return nil
}

// AddTorrent ...
func (tm *TorrentManager) AddTorrent(filePath string) {
	mi, err := metainfo.LoadFromFile(filePath)
	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	ih := spec.InfoHash
	log.Debug("Get torrent from local file", "InfoHash", ih.HexString())

	tm.mu.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Debug("Torrent was already existed. Skip", "InfoHash", ih.HexString())
		tm.mu.Unlock()
		return
	}
	TmpDir := path.Join(tm.TmpDataDir, ih.HexString())
	ExistDir := path.Join(tm.DataDir, ih.HexString())

	useExistDir := false
	if _, err := os.Stat(ExistDir); err == nil {
		log.Info("Seeding from existing file.", "InfoHash", ih.HexString())
		info, err := mi.UnmarshalInfo()
		if err != nil {
			log.Error("error unmarshalling info: ", "info", err)
		}
		if err := verifyTorrent(&info, ExistDir); err != nil {
			log.Warn("torrent failed verification:", "err", err)
		} else {
			useExistDir = true
		}
	}

	if useExistDir {
		spec.Storage = storage.NewFile(ExistDir)

		if len(spec.Trackers) == 0 {
			spec.Trackers = append(spec.Trackers, []string{})
		}
		for _, tracker := range tm.trackers {
			spec.Trackers[0] = append(spec.Trackers[0], tracker)
		}
		t, _, _ := tm.client.AddTorrentSpec(spec)
		var ss []string
		slices.MakeInto(&ss, mi.Nodes)
		tm.client.AddDHTNodes(ss)
		tm.torrents[ih] = &Torrent{
			t,
			defaultBytesLimitation,
			int64(defaultBytesLimitation * expansionFactor),
			0,
			0,
			torrentPending,
		}
		tm.mu.Unlock()
		tm.torrents[ih].Seed()
	} else {
		spec.Storage = storage.NewFile(TmpDir)

		if len(spec.Trackers) == 0 {
			spec.Trackers = append(spec.Trackers, []string{})
		}
		for _, tracker := range tm.trackers {
			spec.Trackers[0] = append(spec.Trackers[0], tracker)
		}
		t, _, _ := tm.client.AddTorrentSpec(spec)
		var ss []string
		slices.MakeInto(&ss, mi.Nodes)
		tm.client.AddDHTNodes(ss)
		tm.torrents[ih] = &Torrent{
			t,
			defaultBytesLimitation,
			int64(defaultBytesLimitation * expansionFactor),
			0,
			0,
			torrentPending,
		}
		tm.mu.Unlock()
		log.Debug("Existing torrent is waiting for gotInfo", "InfoHash", ih.HexString())
		<-t.GotInfo()
		tm.torrents[ih].Run()
	}
}

// AddMagnet ...
func (tm *TorrentManager) AddMagnet(uri string) {
	spec, err := torrent.TorrentSpecFromMagnetURI(uri)
	if err != nil {
		log.Error("Error while adding magnet uri", "Err", err)
	}
	ih := spec.InfoHash
	dataPath := path.Join(tm.TmpDataDir, ih.HexString())
	torrentPath := path.Join(tm.TmpDataDir, ih.HexString(), "torrent")
	seedTorrentPath := path.Join(tm.DataDir, ih.HexString(), "torrent")
	if _, err := os.Stat(torrentPath); err == nil {
		tm.AddTorrent(torrentPath)
		return
	} else if _, err := os.Stat(seedTorrentPath); err == nil {
		tm.AddTorrent(seedTorrentPath)
		return
	}
	log.Debug("Get torrent from magnet uri", "InfoHash", ih.HexString())

	tm.mu.Lock()
	if _, ok := tm.torrents[ih]; ok {
		log.Info("Torrent was already existed. Skip", "InfoHash", ih.HexString())
		tm.mu.Unlock()
		return
	}

	spec.Storage = storage.NewFile(dataPath)
	if len(spec.Trackers) == 0 {
		spec.Trackers = append(spec.Trackers, []string{})
	}
	for _, tracker := range tm.trackers {
		spec.Trackers[0] = append(spec.Trackers[0], tracker)
	}
	t, _, err := tm.client.AddTorrentSpec(spec)
	tm.torrents[ih] = &Torrent{
		t,
		defaultBytesLimitation,
		int64(defaultBytesLimitation * expansionFactor),
		0,
		0,
		torrentPending,
	}
	tm.mu.Unlock()
	log.Debug("Torrent is waiting for gotInfo", "InfoHash", ih.HexString())

	<-t.GotInfo()
	log.Debug("Torrent gotInfo finished", "InfoHash", ih.HexString())
	tm.torrents[ih].Run()

	f, _ := os.Create(torrentPath)
	log.Info("Write torrent file", "InfoHash", ih.HexString(), "path", torrentPath)
	if err := t.Metainfo().Write(f); err != nil {
		log.Error("Error while write torrent file", "error", err)
	}
	defer f.Close()
}

// UpdateMagnet ...
func (tm *TorrentManager) UpdateMagnet(ih metainfo.Hash, BytesRequested int64) {
	log.Info("Update torrent", "InfoHash", ih, "bytes", BytesRequested)
	tm.mu.Lock()
	if t, ok := tm.torrents[ih]; ok {
		t.bytesRequested = BytesRequested
		if t.bytesRequested > t.bytesLimitation {
			t.bytesLimitation = int64(float64(BytesRequested) * expansionFactor)
		}
	}
	tm.mu.Unlock()
}

// DropMagnet ...
func (tm *TorrentManager) DropMagnet(uri string) bool {
	spec, err := torrent.TorrentSpecFromMagnetURI(uri)
	if err != nil {
		log.Warn("error while removing magnet", "error", err)
	}
	ih := spec.InfoHash
	if t, ok := tm.torrents[ih]; ok {
		t.Drop()
		delete(tm.torrents, ih)
		return true
	}
	return false
}

// NewTorrentManager ...
func NewTorrentManager(config *Config) *TorrentManager {
	cfg := torrent.NewDefaultClientConfig()
	cfg.DisableTCP = true
	cfg.DataDir = config.DataDir
	cfg.DisableEncryption = true
	listenAddr := &net.TCPAddr{}
	log.Info("Torrent client listening on", "addr", listenAddr)
	cfg.SetListenAddr(listenAddr.String())
	cfg.Seed = true
	cl, err := torrent.NewClient(cfg)
	if err != nil {
		log.Error("Error while create torrent client", "err", err)
	}

	tmpFilePath := path.Join(config.DataDir, defaultTmpFilePath)
	if _, err := os.Stat(tmpFilePath); err == nil {
		os.Remove(tmpFilePath)
	}
	os.Mkdir(tmpFilePath, os.FileMode(os.ModePerm))

	TorrentManager := &TorrentManager{
		client:        cl,
		torrents:      make(map[metainfo.Hash]*Torrent),
		DataDir:       config.DataDir,
		TmpDataDir:    tmpFilePath,
		closeAll:      make(chan struct{}),
		newTorrent:    make(chan string, newTorrentChanBuffer),
		removeTorrent: make(chan string, removeTorrentChanBuffer),
		updateTorrent: make(chan interface{}, updateTorrentChanBuffer),
	}

	if len(config.DefaultTrackers) > 0 {
		TorrentManager.SetTrackers(strings.Split(config.DefaultTrackers, ","))
	}
	log.Info("Torrent client initialized")

	return TorrentManager
}

func (tm *TorrentManager) Start() error {
	go func() {
		for {
			select {
			case torrent := <-tm.newTorrent:
				if isMagnetURI(torrent) {
					go tm.AddMagnet(torrent)
				} else {
					go tm.AddTorrent(torrent)
				}
			case torrent := <-tm.removeTorrent:
				if isMagnetURI(torrent) {
					go tm.DropMagnet(torrent)
				} else {
				}
			case msg := <-tm.updateTorrent:
				meta := msg.(FlowControlMeta)
				go tm.UpdateMagnet(meta.InfoHash, int64(meta.BytesRequested))
			case <-tm.closeAll:
				tm.halt = true
				tm.client.Close()
				return
			}
		}
	}()

	go func() {
		var counter uint64
		for counter = 0; ; counter++ {
			if tm.halt {
				return
			}
			for ih, t := range tm.torrents {
				if t.Seeding() {
					t.bytesCompleted = t.BytesCompleted()
					t.bytesMissing = t.BytesMissing()
					if counter >= 20 {
						log.Debug("Torrent seeding",
							"InfoHash", ih.HexString(),
							"completed", t.bytesCompleted,
							"total", t.bytesCompleted+t.bytesMissing,
							"seeding", t.Torrent.Seeding(),
						)
					}
				} else if !t.Pending() {
					t.bytesCompleted = t.BytesCompleted()
					t.bytesMissing = t.BytesMissing()
					if t.bytesMissing == 0 {
						os.Symlink(
							path.Join(tm.TmpDataDir, ih.HexString()),
							path.Join(tm.DataDir, ih.HexString()),
						)
						t.Seed()
					} else if t.bytesCompleted >= t.bytesLimitation {
						t.Pause()
					} else if t.bytesCompleted < t.bytesLimitation {
						t.Run()
					}
					if counter >= 20 {
						log.Debug("Torrent progress",
							"InfoHash", ih.HexString(),
							"completed", t.bytesCompleted,
							"requested", t.bytesLimitation,
							"total", t.bytesCompleted+t.bytesMissing,
						)
					}
				} else {
					if counter >= 20 {
						log.Debug("Torrent pending",
							"InfoHash", ih.HexString(),
							"completed", t.bytesCompleted,
							"requested", t.bytesLimitation,
							"total", t.bytesCompleted+t.bytesMissing,
						)
					}
				}
			}
			if counter >= 20 {
				counter = 0
			}
			time.Sleep(time.Second * queryTimeInterval)
		}
	}()

	return nil
}
