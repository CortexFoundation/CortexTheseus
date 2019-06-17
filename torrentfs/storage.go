package torrentfs

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anacrolix/torrent/metainfo"
	"github.com/boltdb/bolt"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
)

const (
	// Chosen to match the usual chunk size in a torrent client. This way,
	// most chunk writes are to exactly one full item in bolt DB.
	chunkSize = 1 << 14
)

type FileInfo struct {
	Meta *FileMeta
	// Transaction hash
	TxHash *common.Hash
	// Contract Address
	ContractAddr *common.Address
	LeftSize     uint64
}

func NewFileInfo(Meta *FileMeta) *FileInfo {
	return &FileInfo{Meta, nil, nil, Meta.RawSize}
}

type MutexCounter int32

func (mc *MutexCounter) Increase() {
	atomic.AddInt32((*int32)(mc), int32(1))
}

func (mc *MutexCounter) Decrease() {
	atomic.AddInt32((*int32)(mc), int32(-1))
}

func (mc *MutexCounter) IsZero() bool {
	return atomic.LoadInt32((*int32)(mc)) == 0
}

type FileStorage struct {
	filesContractAddr map[common.Address]*FileInfo
	db                *bolt.DB

	LastListenBlockNumber uint64

	lock      sync.RWMutex
	bnLock    sync.Mutex
	opCounter MutexCounter
	dataDir   string
	//tmpCache  *lru.Cache
}

var initConfig *Config = nil

func InitConfig() *Config {
	return initConfig
}

func NewFileStorage(config *Config) (*FileStorage, error) {

	if err := os.MkdirAll(config.DataDir, 0700); err != nil {
		return nil, err
	}

	if initConfig == nil {
		initConfig = config
	}

	db, dbErr := bolt.Open(filepath.Join(config.DataDir,
		".file.bolt.db"), 0600, &bolt.Options{
		Timeout: time.Second,
	})
	if dbErr != nil {
		return nil, dbErr
	}
	db.NoSync = true

	fs := &FileStorage{
		// filesInfoHash:     make(map[metainfo.Hash]*FileInfo),
		filesContractAddr: make(map[common.Address]*FileInfo),
		db:                db,
		opCounter:         0,
		dataDir:           config.DataDir,
	}
	fs.readBlockNumber()
	//tmpCache, _ := lru.New(120)

	return fs, nil
}

func (fs *FileStorage) AddFile(x *FileInfo) error {
	addr := *x.ContractAddr
	if _, ok := fs.filesContractAddr[addr]; ok {
		return errors.New("file already existed")
	}
	fs.filesContractAddr[addr] = x
	return nil
}

func (fs *FileStorage) GetFileByAddr(addr common.Address) *FileInfo {
	if f, ok := fs.filesContractAddr[addr]; ok {
		return f
	}
	return nil
}

func Exist(infohash string) bool {
	ih := metainfo.NewHashFromHex(infohash[2:])
	tm := CurrentTorrentManager
	torrent, ok := tm.torrents[ih]
	if !ok {
		return false
	}
	return torrent.IsAvailable()
}

func Available(infohash string, rawSize int64) bool {
	ih := metainfo.NewHashFromHex(infohash[2:])
	tm := CurrentTorrentManager
	torrent, ok := tm.torrents[ih]
	if !ok {
		return false
	}
	return torrent.IsAvailable()
}


func ExistTorrent(infohash string) bool {
	ih := metainfo.NewHashFromHex(infohash[2:])
	tm := CurrentTorrentManager
	torrent, ok := tm.torrents[ih]
	if !ok {
		return false
	}
	return torrent.HasTorrent()
}

func (fs *FileStorage) Close() error {
	log.Info("Torrent File Storage Closed", "database", fs.db.Path())

	// Wait for file storage closed...
	for {
		if fs.opCounter.IsZero() {
			// persist storage block number
			fs.writeBlockNumber()
			return fs.db.Close()
		}

		// log.Debug("Waiting for boltdb operating...")
		time.Sleep(time.Microsecond)
	}
}

var (
	ErrReadDataFromBoltDB = errors.New("Bolt DB Read Error")
)

func (fs *FileStorage) GetBlockByNumber(blockNum uint64) *Block {
	var block Block

	fs.opCounter.Increase()
	defer fs.opCounter.Decrease()

	cb := func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte("blocks"))
		if buk == nil {
			return ErrReadDataFromBoltDB
		}
		k, err := json.Marshal(blockNum)
		if err != nil {
			return ErrReadDataFromBoltDB
		}

		fs.lock.RLock()
		v := buk.Get(k)
		fs.lock.RUnlock()

		if v == nil {
			return ErrReadDataFromBoltDB
		}
		if err := json.Unmarshal(v, &block); err != nil {
			return err
		}

		return nil
	}

	if err := fs.db.View(cb); err != nil {
		return nil
	}
	return &block
}

func (fs *FileStorage) WriteBlock(b *Block) error {
	fs.opCounter.Increase()
	defer fs.opCounter.Decrease()

	err := fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("blocks"))
		if err != nil {
			return err
		}
		v, err := json.Marshal(b)
		if err != nil {
			return err
		}
		k, err := json.Marshal(b.Number)
		if err != nil {
			return err
		}

		fs.lock.Lock()
		e := buk.Put(k, v)
		fs.lock.Unlock()

		return e
	})

	//if err == nil && b.Number > fs.LastListenBlockNumber {
	if err == nil {
		fs.bnLock.Lock()
		fs.LastListenBlockNumber = b.Number
		fs.writeBlockNumber()
		fs.bnLock.Unlock()
	}

	return err
}

func (fs *FileStorage) readBlockNumber() error {
	return fs.db.View(func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte("currentBlockNumber"))
		if buk == nil {
			return ErrReadDataFromBoltDB
		}

		v := buk.Get([]byte("key"))

		if v == nil {
			return ErrReadDataFromBoltDB
		}

		number, err := strconv.ParseUint(string(v), 16, 64)
		if err != nil {
			return err
		}

		fs.LastListenBlockNumber = number

		return nil
	})
}

func (fs *FileStorage) writeBlockNumber() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("currentBlockNumber"))
		if err != nil {
			return err
		}

		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.LastListenBlockNumber, 16)))

		return e
	})
}

type FlowControlMeta struct {
	InfoHash       metainfo.Hash
	BytesRequested uint64
}
