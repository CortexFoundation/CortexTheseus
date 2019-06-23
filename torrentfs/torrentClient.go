package torrentfs

import (
  "bytes"
  "errors"
  "crypto/sha1"
  "io/ioutil"
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
  "sort"
  "math/rand"

  "github.com/CortexFoundation/CortexTheseus/log"
  "github.com/CortexFoundation/CortexTheseus/params"
  "github.com/anacrolix/torrent"
  "github.com/anacrolix/torrent/metainfo"
  "github.com/anacrolix/torrent/mmap_span"
  "github.com/anacrolix/torrent/storage"

  "github.com/anacrolix/dht"
)

const (
  defaultBytesLimitation          = 512 * 1024
  queryTimeInterval               = 3
  defaultSeedInterval             = 600
  removeTorrentChanBuffer         = 16
  newTorrentChanBuffer            = 32
  updateTorrentChanBuffer         = 32
  expansionFactor         float64 = 1.2
  torrentPending                  = 0
  torrentPaused                   = 1
  torrentRunning                  = 2
  torrentSeeding                  = 3
  torrentSeedingInQueue           = 4
  defaultTmpFilePath              = ".tmp"
)

// Torrent ...
type Torrent struct {
  *torrent.Torrent
  maxEstablishedConns int
  currentConns        int
  bytesRequested      int64
  bytesLimitation     int64
  bytesCompleted      int64
  bytesMissing        int64
  status              int
  infohash            string
  filepath            string
  cited               int64
  weight              int
  loop                int
}

func (t *Torrent) BytesLeft() int64 {
  return t.bytesRequested - t.bytesCompleted
}

func (t *Torrent) InfoHash() string {
  return t.infohash
}

func (t *Torrent) GetFile(subpath string) ([]byte, error) {
  if !t.IsAvailable() {
    return nil,  errors.New(fmt.Sprintf("InfoHash %s not Available", t.infohash))
  }
  filepath := path.Join(t.filepath, subpath)
  // fmt.Println("modelCfg = ", modelCfg)
  if _, cfgErr := os.Stat(filepath); os.IsNotExist(cfgErr) {
    return nil, errors.New(fmt.Sprintf("File %s not Available", filepath))
  }
  data, data_err := ioutil.ReadFile(filepath)
  return data, data_err
}

func (t *Torrent) IsAvailable() bool {
  t.cited += 1
  if (t.status == torrentSeeding || t.status == torrentSeedingInQueue) {
    return true
  }
  return false
}

func (t *Torrent) HasTorrent() bool {
  return t.status != torrentPending
}

func (t *Torrent) WriteTorrent() {
  // log.Debug("Torrent gotInfo finished")
  f, _ := os.Create(path.Join(t.filepath, "torrent"))
  log.Debug("Write torrent file", "path", t.filepath)
  if err := t.Metainfo().Write(f); err != nil {
    log.Error("Error while write torrent file", "error", err)
  }

  defer f.Close()
  t.status = torrentPaused
}

func (t *Torrent) SeedInQueue() {  
  if t.currentConns != 0 {
	t.currentConns = 0
    t.Torrent.SetMaxEstablishedConns(0)
  }
  t.status = torrentSeedingInQueue
  t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
}

func (t *Torrent) Seed() {
  t.status = torrentSeeding
  if t.currentConns == 0 {
	t.currentConns = t.maxEstablishedConns
	t.Torrent.SetMaxEstablishedConns(t.currentConns)
  }
  t.Torrent.DownloadAll()
}

func (t *Torrent) Seeding() bool {
  return t.status == torrentSeeding || 
           t.status == torrentSeedingInQueue
}

// Pause ...
func (t *Torrent) Pause() {
  if t.currentConns != 0 {
	t.currentConns = 0
    t.Torrent.SetMaxEstablishedConns(0)
  }
  if t.status != torrentPaused {
    t.status = torrentPaused
    t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
  }
}

// Paused ...
func (t *Torrent) Paused() bool {
  return t.status == torrentPaused
}

func (t *Torrent) Length() int64 {
  return t.bytesCompleted + t.bytesMissing
}

func (t *Torrent) NumPieces() int {
  return t.Torrent.NumPieces()
}

// Run ...
func (t *Torrent) Run() {
  loop := int((t.bytesRequested * int64(t.NumPieces()) + t.Length() - 1) / t.Length())
  if loop > t.NumPieces() {
    loop = t.NumPieces()
  }
  if t.currentConns == 0 {
	t.currentConns = t.maxEstablishedConns
	t.Torrent.SetMaxEstablishedConns(t.currentConns)
  }
  t.status = torrentRunning
  if loop > t.loop {
    t.loop = loop
    t.Torrent.DownloadPieces(0, loop)
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
  client            *torrent.Client
  torrents          map[metainfo.Hash]*Torrent
  seedingTorrents   map[metainfo.Hash]*Torrent
  activeTorrents    map[metainfo.Hash]*Torrent
  pendingTorrents   map[metainfo.Hash]*Torrent
  maxSeedTask       int
  maxEstablishedConns int
  maxActiveTask     int
  trackers          [][]string
  DataDir           string
  TmpDataDir        string
  closeAll          chan struct{}
  newTorrent        chan interface{}
  removeTorrent     chan metainfo.Hash
  updateTorrent     chan interface{}
  halt              bool
  mu                sync.Mutex
  lock              sync.RWMutex
}

func (tm *TorrentManager) CreateTorrent(t *torrent.Torrent, requested int64, status int, ih metainfo.Hash) *Torrent {
  tt := &Torrent{
    t,
    tm.maxEstablishedConns, tm.maxEstablishedConns, 
    requested, 
    int64(float64(requested) * expansionFactor), 
    0, 0, status, 
    ih.String(), 
    path.Join(tm.TmpDataDir, ih.String()), 
    0, 1, 0,
  }
  tm.SetTorrent(ih, tt)
  return tt
}

func (tm *TorrentManager) GetTorrent(ih metainfo.Hash) *Torrent {
  tm.lock.RLock()
  defer tm.lock.RUnlock()
  torrent, ok := tm.torrents[ih]
  if !ok {
    return nil
  }
  return torrent
}

func (tm *TorrentManager) SetTorrent(ih metainfo.Hash, torrent *Torrent) {
  tm.lock.Lock()
  defer tm.lock.Unlock()
  tm.torrents[ih] = torrent
  tm.pendingTorrents[ih] = torrent
}

func (tm *TorrentManager) Close() error {
  close(tm.closeAll)
  log.Info("Torrent Download Manager Closed")
  return nil
}

func (tm *TorrentManager) NewTorrent(input interface{}) error {
//  fmt.Println("NewTorrent", input.(FlowControlMeta))
  tm.newTorrent <- input
  return nil
}

func (tm *TorrentManager) RemoveTorrent(input metainfo.Hash) error {
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

func GetMagnetURI(infohash metainfo.Hash) string {
  return "magnet:?xt=urn:btih:" + infohash.String()
}

func (tm *TorrentManager) SetTrackers(trackers []string) {
  tm.trackers = append(tm.trackers, trackers)
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

func (tm *TorrentManager) AddTorrent(filePath string, BytesRequested int64) {
  if _, err := os.Stat(filePath); err != nil {
    return
  }
  mi, err := metainfo.LoadFromFile(filePath)
  if err != nil {
    log.Error("Error while adding torrent", "Err", err)
    return
  }
  spec := torrent.TorrentSpecFromMetaInfo(mi)
  ih := spec.InfoHash
  log.Trace("Get torrent from local file", "InfoHash", ih.HexString())

  if tm.GetTorrent(ih) != nil {
    log.Trace("Torrent was already existed. Skip", "InfoHash", ih.HexString())
    return
  }
  TmpDir := path.Join(tm.TmpDataDir, ih.HexString())
  ExistDir := path.Join(tm.DataDir, ih.HexString())

  useExistDir := false
  if _, err := os.Stat(ExistDir); err == nil {
    log.Debug("Seeding from existing file.", "InfoHash", ih.HexString())
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
    log.Trace("existing dir", "dir", ExistDir)
    spec.Storage = storage.NewFile(ExistDir)
    for _, tracker := range tm.trackers {
      spec.Trackers = append(spec.Trackers, tracker)
    }
    t, _, _ := tm.client.AddTorrentSpec(spec)
    var ss []string
    slices.MakeInto(&ss, mi.Nodes)
    tm.client.AddDHTNodes(ss)
    torrent := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
    <-t.GotInfo()
	torrent.SeedInQueue()
  } else {
    spec.Storage = storage.NewFile(TmpDir)
    for _, tracker := range tm.trackers {
      spec.Trackers = append(spec.Trackers, tracker)
    }
    t, _, _ := tm.client.AddTorrentSpec(spec)
    var ss []string
    slices.MakeInto(&ss, mi.Nodes)
    tm.client.AddDHTNodes(ss)
    torrent := tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
    <-t.GotInfo()
	torrent.Pause()
  }
}

func (tm *TorrentManager) AddInfoHash(ih metainfo.Hash, BytesRequested int64) {
  if tm.GetTorrent(ih) != nil {
    tm.UpdateInfoHash(ih, BytesRequested)
    return
  }

  dataPath := path.Join(tm.TmpDataDir, ih.HexString())
  torrentPath := path.Join(tm.TmpDataDir, ih.HexString(), "torrent")
  seedTorrentPath := path.Join(tm.DataDir, ih.HexString(), "torrent")
  
  if _, err := os.Stat(seedTorrentPath); err == nil {
    tm.AddTorrent(seedTorrentPath, BytesRequested)
    return
  } else if _, err := os.Stat(torrentPath); err == nil {
    tm.AddTorrent(torrentPath, BytesRequested)
    return
  }
  log.Debug("Get torrent from infohash", "InfoHash", ih.HexString())

  if tm.GetTorrent(ih) != nil {
    log.Warn("Torrent was already existed. Skip", "InfoHash", ih.HexString())
    //tm.mu.Unlock()
    return
  }
  
  spec := &torrent.TorrentSpec{
    Trackers: [][]string{},
    DisplayName: ih.String(),
    InfoHash: ih,
    Storage: storage.NewFile(dataPath),
  }

  for _, tracker := range tm.trackers {
    spec.Trackers = append(spec.Trackers, tracker)
  }
  log.Trace("Torrent specific info", "spec", spec)

  t, _, _ := tm.client.AddTorrentSpec(spec)
  tm.CreateTorrent(t, BytesRequested, torrentPending, ih)
  //tm.mu.Unlock()
  log.Trace("Torrent is waiting for gotInfo", "InfoHash", ih.HexString())
}

// UpdateInfoHash ...
func (tm *TorrentManager) UpdateInfoHash(ih metainfo.Hash, BytesRequested int64) {
  log.Debug("Update torrent", "InfoHash", ih, "bytes", BytesRequested)
  if t := tm.GetTorrent(ih); t != nil {
    if BytesRequested < t.bytesRequested {
      return
    }
    t.bytesRequested = BytesRequested
    if t.bytesRequested > t.bytesLimitation {
      t.bytesLimitation = int64(float64(BytesRequested) * expansionFactor)
    }
  }
  //tm.mu.Unlock()
}

// DropMagnet ...
func (tm *TorrentManager) DropMagnet(ih metainfo.Hash) bool {
  if t := tm.GetTorrent(ih); t != nil {
    t.Torrent.Drop()
    tm.lock.Lock()
    delete(tm.torrents, ih)
    tm.lock.Unlock()
    return true
  }
  return false
}

var CurrentTorrentManager *TorrentManager = nil
// NewTorrentManager ...
func NewTorrentManager(config *Config) *TorrentManager {
  cfg := torrent.NewDefaultClientConfig()
  cfg.DisableUTP = config.DisableUTP
  cfg.NoDHT = false
  cfg.DhtStartingNodes = dht.GlobalBootstrapAddrs
  cfg.DataDir = config.DataDir
  cfg.DisableEncryption = true
  cfg.ExtendedHandshakeClientVersion = params.VersionWithMeta
  listenAddr := &net.TCPAddr{}
  log.Info("Torrent client listening on", "addr", listenAddr)
  //cfg.SetListenAddr(listenAddr.String())
  cfg.HTTPUserAgent = "Cortex"
  cfg.Seed = true
  cfg.EstablishedConnsPerTorrent = 10
  cfg.HalfOpenConnsPerTorrent = 5
  cfg.DropDuplicatePeerIds = true
  log.Info("Torrent client configuration", "config", cfg)
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
    client:               cl,
    torrents:             make(map[metainfo.Hash]*Torrent),
    pendingTorrents:      make(map[metainfo.Hash]*Torrent),
    seedingTorrents:      make(map[metainfo.Hash]*Torrent),
    activeTorrents:       make(map[metainfo.Hash]*Torrent),
    maxSeedTask:           config.MaxSeedingNum,
    maxActiveTask:         config.MaxActiveNum,
    maxEstablishedConns:   cfg.EstablishedConnsPerTorrent,
    DataDir:              config.DataDir,
    TmpDataDir:           tmpFilePath,
    closeAll:             make(chan struct{}),
    newTorrent:           make(chan interface{}, newTorrentChanBuffer),
    removeTorrent:        make(chan metainfo.Hash, removeTorrentChanBuffer),
    updateTorrent:        make(chan interface{}, updateTorrentChanBuffer),
  }

  if len(config.DefaultTrackers) > 0 {
    log.Info("Tracker list", "trackers", config.DefaultTrackers)
    TorrentManager.SetTrackers(config.DefaultTrackers)
  }
  log.Info("Torrent client initialized")

  CurrentTorrentManager = TorrentManager
  return TorrentManager
}

func (tm *TorrentManager) Start() error {

  go tm.mainLoop()
  go tm.listenTorrentProgress()

  return nil
}

func (tm *TorrentManager) mainLoop() {
  for {
    select {
    case msg := <-tm.newTorrent:
      meta := msg.(FlowControlMeta)
      log.Debug("TorrentManager", "newTorrent", meta.InfoHash.String())
      go tm.AddInfoHash(meta.InfoHash, int64(meta.BytesRequested))
    case torrent := <-tm.removeTorrent:
      go tm.DropMagnet(torrent)
    case msg := <-tm.updateTorrent:
      meta := msg.(FlowControlMeta)
      go tm.UpdateInfoHash(meta.InfoHash, int64(meta.BytesRequested))
    case <-tm.closeAll:
      tm.halt = true
      tm.client.Close()
      return
    }
  }
}

const (
  loops = 2
)

type ActiveTorrentList []*Torrent
func (s ActiveTorrentList) Len() int { return len(s) }
func (s ActiveTorrentList) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s ActiveTorrentList) Less(i, j int) bool { return s[i].BytesLeft() > s[j].BytesLeft() || (s[i].BytesLeft() == s[j].BytesLeft() && s[i].weight > s[j].weight) }

type seedingTorrentList []*Torrent
func (s seedingTorrentList) Len() int { return len(s) }
func (s seedingTorrentList) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s seedingTorrentList) Less(i, j int) bool { return s[i].weight > s[j].weight }

func (tm *TorrentManager) listenTorrentProgress() {
  var counter uint64
  for counter = 0; ; counter++ {
    if tm.halt {
      return
    }
  
    tm.lock.RLock()

    var maxCited int64 = 1

    for _, t := range tm.torrents {
      if t.cited > maxCited {
        maxCited = t.cited
      }
    }
    
    for _, t := range tm.torrents {
      t.weight = 1 + int(t.cited * 10 / maxCited)
    }

    for ih, t := range tm.pendingTorrents {
      t.loop += 1
      if t.Seeding() {
        delete(tm.pendingTorrents, ih)
        tm.seedingTorrents[ih] = t
        t.loop = 0
      } else if !t.Pending() {
        delete(tm.pendingTorrents, ih)
        tm.activeTorrents[ih] = t
        t.loop = 0
      } else if t.Torrent.Info() != nil {
        t.WriteTorrent()
      }
    }
    
    var activeTorrents []*Torrent
    for ih, t := range tm.activeTorrents {
      t.bytesCompleted = t.BytesCompleted()
      t.bytesMissing = t.BytesMissing()
      if t.bytesMissing == 0 {
        os.Symlink(
          path.Join(defaultTmpFilePath, t.InfoHash()),
          path.Join(tm.DataDir, t.InfoHash()),
        )
        delete(tm.activeTorrents, ih)
        tm.seedingTorrents[ih] = t
        t.status = torrentSeeding
        t.loop = defaultSeedInterval / queryTimeInterval
        continue
      }
    
      if t.bytesCompleted >= t.bytesLimitation {
        t.Pause()
      } else if t.bytesCompleted < t.bytesRequested {
        activeTorrents = append(activeTorrents, t)
      }
    }

    if len(activeTorrents) <= tm.maxActiveTask {
      for _, t := range activeTorrents {
        t.Run()
      }
    } else {
      sort.Stable(ActiveTorrentList(activeTorrents))
      for i := 0; i < tm.maxActiveTask; i++ {
        activeTorrents[i].Run()
      }
      for i := tm.maxActiveTask; i < len(activeTorrents); i++ {
        activeTorrents[i].Pause()
      }
    }

    if len(tm.seedingTorrents) <= tm.maxSeedTask {
      for _, t := range tm.seedingTorrents {
        t.Seed()
        t.loop = 0
      }
    } else {
      var totalWeight int = 0
      var nSeedTask int = tm.maxSeedTask
      for _, t := range tm.seedingTorrents {
        if t.loop == 0 { 
          totalWeight += t.weight  
        } else if t.status == torrentSeeding {
          nSeedTask -= 1
        }
      }

      for _, t := range tm.seedingTorrents {
        if t.loop > 0 {
          t.loop -= 1
        } else {
          t.loop = defaultSeedInterval / queryTimeInterval
          prob := float32(t.weight) * float32(nSeedTask) / float32(totalWeight)
          if rand.Float32() < prob {
            t.Seed()
          } else {
            t.SeedInQueue()
          }
        }
      }
    }

    if counter >= loops {
      for _, t := range tm.activeTorrents {
        log.Trace("Torrent progress",
          "InfoHash", t.InfoHash(),
          "completed", t.bytesCompleted,
          "requested", t.bytesLimitation,
          "total", t.bytesCompleted+t.bytesMissing,
          "status", t.status)
      }
      var nSeed int = 0
      for _, t := range tm.seedingTorrents {
        if t.status == torrentSeeding {
          log.Trace("Torrent seeding",
            "InfoHash", t.InfoHash(),
            "completed", t.bytesCompleted,
            "requested", t.bytesLimitation,
            "total", t.bytesCompleted+t.bytesMissing,
            "status", t.status)
          nSeed += 1
        }
      }

      log.Info("TorrentFs working status", "pending", len(tm.pendingTorrents), "active", len(tm.activeTorrents), "seeding", nSeed, "seeding_in_queue", len(tm.seedingTorrents) - nSeed)
      counter = 0
    }

    tm.lock.RUnlock()
    time.Sleep(time.Second * queryTimeInterval)
  }
}
