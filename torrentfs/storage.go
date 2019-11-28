package torrentfs

import (
	"encoding/json"
	"errors"
	//"fmt"
	"os"
	"path/filepath"
	//"path"
	//"io/ioutil"
	"strconv"
	//"strings"
	//"sync"
	//"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/anacrolix/torrent/metainfo"
	bolt "github.com/etcd-io/bbolt"
)

const (
// Chosen to match the usual chunk size in a torrent client. This way,
// most chunk writes are to exactly one full item in bolt DB.
//chunkSize = 1 << 14
)

type FileInfo struct {
	Meta *FileMeta
	// Transaction hash
	TxHash *common.Hash
	// Contract Address
	ContractAddr *common.Address
	LeftSize     uint64
	//Index        uint64
}

//type MutexCounter int32

/*func (mc *MutexCounter) Increase() {
	atomic.AddInt32((*int32)(mc), int32(1))
}

func (mc *MutexCounter) Decrease() {
	atomic.AddInt32((*int32)(mc), int32(-1))
}

func (mc *MutexCounter) IsZero() bool {
	return atomic.LoadInt32((*int32)(mc)) == 0
}*/

type FileStorage struct {
	filesContractAddr map[common.Address]*FileInfo
	files             []*FileInfo
	blocks            []*Block
	db                *bolt.DB

	LastListenBlockNumber uint64
	CheckPoint            uint64
	//LastFileIndex         uint64

	//lock      sync.RWMutex
	//bnLock    sync.Mutex
	//opCounter MutexCounter
	dataDir string
	//tmpCache  *lru.Cache
	//indexLock sync.RWMutex
}

var initConfig *Config = nil

//var TorrentAPIAvailable sync.Mutex

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
	//db.NoSync = true

	fs := &FileStorage{
		// filesInfoHash:     make(map[metainfo.Hash]*FileInfo),
		filesContractAddr: make(map[common.Address]*FileInfo),
		db:                db,
		//opCounter:         0,
		dataDir: config.DataDir,
	}

	fs.initBlockNumber()
	fs.initCheckPoint()
	fs.initBlocks()
	//fs.readLastFileIndex()
	fs.initFiles()
	//tmpCache, _ := lru.New(120)

	return fs, nil
}

func (fs *FileStorage) Files() []*FileInfo {
	return fs.files
}

func (fs *FileStorage) Blocks() []*Block {
	return fs.blocks
}

func (fs *FileStorage) NewFileInfo(Meta *FileMeta) *FileInfo {
	//ret := &FileInfo{Meta, nil, nil, Meta.RawSize, 0}
	ret := &FileInfo{Meta, nil, nil, Meta.RawSize}
	return ret
}

/*func (fs *FileStorage) AddCachedFile(x *FileInfo) error {
	addr := *x.ContractAddr
	fs.filesContractAddr[addr] = x
	fs.files = append(fs.files, x)
	return nil
}*/

//func (fs *FileStorage) CurrentTorrentManager() *TorrentManager {
//	return CurrentTorrentManager
//}

func (fs *FileStorage) AddFile(x *FileInfo) (uint64, error) {
	addr := *x.ContractAddr
	if _, ok := fs.filesContractAddr[addr]; ok {
		return 0, nil
	}

	fs.filesContractAddr[addr] = x

	update, err := fs.WriteFile(x)
	if err != nil {
		return 0, err
	}

	if !update {
		return 0, nil
	}
	return 1, nil
}

func (fs *FileStorage) GetFileByAddr(addr common.Address) *FileInfo {
	if f, ok := fs.filesContractAddr[addr]; ok {
		return f
	}
	return nil
}

// func Exist(infohash string) bool {
// 	TorrentAPIAvailable.Lock()
// 	defer TorrentAPIAvailable.Unlock()
// 	ih := metainfo.NewHashFromHex(infohash[2:])
// 	tm := CurrentTorrentManager
// 	if torrent := tm.GetTorrent(ih); torrent == nil {
// 		return false
// 	} else {
// 		return torrent.IsAvailable()
// 	}
// }

/*func Available(infohash string, rawSize int64) (bool, error) {
	//log.Info("Available", "infohash", infohash, "RawSize", rawSize)
	//TorrentAPIAvailable.Lock()
	//defer TorrentAPIAvailable.Unlock()
	//if !strings.HasPrefix(infohash, "0x") {
	//	return false, errors.New("invalid info hash format")
	//}
	ih := metainfo.NewHashFromHex(infohash)
	tm := CurrentTorrentManager
	//log.Debug("storage", "ih", ih)
	if torrent := tm.GetTorrent(ih); torrent == nil {
		//log.Debug("storage", "ih", ih, "torrent", torrent)
		log.Info("Torrent not found", "hash", infohash)
		return false, errors.New("download not completed")
	} else {
		if !torrent.IsAvailable() {
			log.Warn("[Not available] Download not completed", "hash", infohash, "raw", rawSize, "complete", torrent.BytesCompleted())
			return false, errors.New(fmt.Sprintf("download not completed: %d %d", torrent.BytesCompleted(), rawSize))
		}
		//log.Debug("storage", "Available", torrent.IsAvailable(), "torrent.BytesCompleted()", torrent.BytesCompleted(), "rawSize", rawSize)
		//log.Info("download not completed", "complete", torrent.BytesCompleted(), "miss", torrent.BytesMissing(), "raw", rawSize)
		return torrent.BytesCompleted() <= rawSize, nil
	}
}*/

/*func LoadFile(infohash string, fn string) ([]byte, error) {
        data, err := ioutil.ReadFile(fn)
        return data, err
}*/

/*func GetFile(infohash string, path string) ([]byte, error) {
	infohash = strings.ToLower(infohash)
	//TorrentAPIAvailable.Lock()
	//defer TorrentAPIAvailable.Unlock()
	if strings.HasPrefix(infohash, "0x") {
		//infohash = infohash[2:]
		infohash = infohash[2:]
	}
	ih := metainfo.NewHashFromHex(infohash)
	tm := CurrentTorrentManager
	var torrent Torrent
	if torrent := tm.GetTorrent(ih); torrent == nil {
		log.Debug("storage", "ih", ih, "torrent", torrent)
		fmt.Println("torrent", torrent)
		return nil, errors.New("Torrent not Available: " + infohash)
	}
	data, err := torrent.GetFile(path)
	return data, err
}*/

/*func ExistTorrent(infohash string) bool {
	//TorrentAPIAvailable.Lock()
	//defer TorrentAPIAvailable.Unlock()
	ih := metainfo.NewHashFromHex(infohash[2:])
	tm := CurrentTorrentManager
	if torrent := tm.GetTorrent(ih); torrent == nil {
		return false
	} else {
		return torrent.HasTorrent()
	}
}*/

func (fs *FileStorage) Close() error {
	defer fs.db.Close()
	// Wait for file storage closed...
	//for {
	//	if fs.opCounter.IsZero() {
	// persist storage block number
	fs.writeCheckPoint()
	log.Info("Torrent File Storage Closed", "database", fs.db.Path())
	return fs.writeBlockNumber()
	//fs.writeLastFileIndex()
	//	}

	// log.Debug("Waiting for boltdb operating...")
	//	time.Sleep(time.Microsecond)
	//}
}

var (
	ErrReadDataFromBoltDB = errors.New("bolt DB Read Error")
)

/*func (fs *FileStorage) GetFileByNumber(index uint64) *FileInfo {
	var info FileInfo

	fs.opCounter.Increase()
	defer fs.opCounter.Decrease()

	cb := func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte("files"))
		if buk == nil {
			return ErrReadDataFromBoltDB
		}
		k, err := json.Marshal(index)
		if err != nil {
			return ErrReadDataFromBoltDB
		}

		fs.lock.RLock()
		v := buk.Get(k)
		fs.lock.RUnlock()

		if v == nil {
			return ErrReadDataFromBoltDB
		}
		if err := json.Unmarshal(v, &info); err != nil {
			return err
		}

		return nil
	}

	if err := fs.db.View(cb); err != nil {
		return nil
	}
	log.Debug("Read fileinfo from database", "info", info, "meta", info.Meta)
	return &info
}*/

func (fs *FileStorage) GetBlockByNumber(blockNum uint64) *Block {
	var block Block

	cb := func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte("blocks"))
		if buk == nil {
			return ErrReadDataFromBoltDB
		}
		k, err := json.Marshal(blockNum)
		if err != nil {
			return ErrReadDataFromBoltDB
		}

		v := buk.Get(k)

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

func (fs *FileStorage) WriteFile(f *FileInfo) (bool, error) {
	//fs.opCounter.Increase()
	//defer fs.opCounter.Decrease()
	update := false
	err := fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("files"))
		if err != nil {
			return err
		}
		v, err := json.Marshal(f)
		if err != nil {
			return err
		}
		k, err := json.Marshal(f.Meta.InfoHash)
		if err != nil {
			return err
		}

		bef := buk.Get(k)
		if bef == nil {
			update = true
			return buk.Put(k, v)
		} else {
			var info FileInfo
			if err := json.Unmarshal(bef, &info); err != nil {
				update = true
				return buk.Put(k, v)
			}

			if info.LeftSize > f.LeftSize {
				update = true
				return buk.Put(k, v)
			} else {
				log.Debug("Write same file in 2 address", "hash", info.Meta.InfoHash.String(), "old", info.LeftSize, "new", f.LeftSize)
			}
		}
		return nil
	})

	return update, err
}

func (fs *FileStorage) WriteBlock(b *Block, record bool) error {
	if record {
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

			e := buk.Put(k, v)

			return e
		})

		if err != nil {
			return err
		}

		if b.Number > fs.CheckPoint {
			fs.CheckPoint = b.Number
			fs.writeCheckPoint()
		}
	}

	if b.Number < fs.LastListenBlockNumber {
		return nil
	}

	fs.LastListenBlockNumber = b.Number
	return fs.writeBlockNumber()
}

func (fs *FileStorage) initBlocks() error {
	return fs.db.View(func(tx *bolt.Tx) error {
		if buk := tx.Bucket([]byte("blocks")); buk == nil {
			return ErrReadDataFromBoltDB
		} else {
			c := buk.Cursor()

			for k, v := c.First(); k != nil; k, v = c.Next() {

				var x Block

				if err := json.Unmarshal(v, &x); err != nil {
					return err
				}
				fs.blocks = append(fs.blocks, &x)
			}
			log.Info("Fs blocks initializing ... ...", "blocks", len(fs.blocks))
			return nil
		}
	})
}

func (fs *FileStorage) initFiles() error {
	return fs.db.View(func(tx *bolt.Tx) error {
		if buk := tx.Bucket([]byte("files")); buk == nil {
			return ErrReadDataFromBoltDB
		} else {
			c := buk.Cursor()

			for k, v := c.First(); k != nil; k, v = c.Next() {

				var x FileInfo

				if err := json.Unmarshal(v, &x); err != nil {
					return err
				}
				fs.filesContractAddr[*x.ContractAddr] = &x
				fs.files = append(fs.files, &x)
			}
			return nil
		}
	})
}

/*func (fs *FileStorage) readLastFileIndex() error {
	return fs.db.View(func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte("lastFileIndex"))
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

		//fs.indexLock.Lock()
		//defer fs.indexLock.Unlock()
		fs.LastFileIndex = number

		return nil
	})
}*/

/*func (fs *FileStorage) writeLastFileIndex() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("lastFileIndex"))
		if err != nil {
			return err
		}
		//fs.lock.Lock()
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.LastFileIndex, 16)))
		//fs.lock.Unlock()

		return e
	})
}*/
func (fs *FileStorage) initCheckPoint() error {
	return fs.db.View(func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte("checkpoint"))
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

		fs.CheckPoint = number

		return nil
	})
}

func (fs *FileStorage) initBlockNumber() error {
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

func (fs *FileStorage) writeCheckPoint() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("checkpoint"))
		if err != nil {
			return err
		}
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.CheckPoint, 16)))

		return e
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
	IsCreate       bool
}
