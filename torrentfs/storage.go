package torrentfs

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/params"
	//"fmt"
	"github.com/pborman/uuid"
	"os"
	"path/filepath"
	//"path"
	"sort"
	//"io/ioutil"
	"strconv"
	//"strings"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	//"sync"
	//"sync/atomic"
	"time"

	"crypto/sha256"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/anacrolix/torrent/metainfo"
	bolt "github.com/etcd-io/bbolt"
	//"math/rand"
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
	files             []*FileInfo //only storage init files from local storage
	blocks            []*Block    //only storage init ckp blocks from local storage
	db                *bolt.DB
	version           string

	id                    uint64
	CheckPoint            uint64
	LastListenBlockNumber uint64

	//elements []*BlockContent
	leaves []Content
	tree   *MerkleTree
	//LastFileIndex         uint64

	//lock      sync.RWMutex
	//bnLock    sync.Mutex
	//opCounter MutexCounter
	dataDir string
	//tmpCache  *lru.Cache
	//indexLock sync.RWMutex
	config *Config
}

//var initConfig *Config = nil

func NewFileStorage(config *Config) (*FileStorage, error) {

	if err := os.MkdirAll(config.DataDir, 0700); err != nil {
		return nil, err
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

	fs.config = config

	fs.version = version

	if err := fs.initBlockNumber(); err != nil {
		return nil, err
	}
	if err := fs.initCheckPoint(); err != nil {
		return nil, err
	}
	if err := fs.initBlocks(); err != nil {
		return nil, err
	}
	if err := fs.initFiles(); err != nil {
		return nil, err
	}
	if err := fs.initMerkleTree(); err != nil {
		return nil, err
	}

	fs.initFsId()

	log.Info("Storage ID init", "id", fs.id)

	return fs, nil
}

type BlockContent struct {
	x string
}

func (t BlockContent) CalculateHash() ([]byte, error) {
	h := sha256.New()
	if _, err := h.Write([]byte(t.x)); err != nil {
		return nil, err
	}

	return h.Sum(nil), nil
}

//Equals tests for equality of two Contents
func (t BlockContent) Equals(other Content) (bool, error) {
	return t.x == other.(BlockContent).x, nil
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

func (fs *FileStorage) initMerkleTree() error {
	fs.leaves = append(fs.leaves, BlockContent{x: params.MainnetGenesisHash.String()}) //"0x21d6ce908e2d1464bd74bbdbf7249845493cc1ba10460758169b978e187762c1"})
	tr, err := NewTree(fs.leaves)
	if err != nil {
		return err
	}
	fs.tree = tr
	for _, block := range fs.blocks {
		if err := fs.addLeaf(block); err != nil {
			panic("Storage merkletree construct failed")
		}
	}

	log.Info("Storage merkletree initialization", "root", hexutil.Encode(fs.tree.MerkleRoot()))

	return nil
}

func (fs *FileStorage) addLeaf(block *Block) error {
	fs.leaves = append(fs.leaves, BlockContent{x: block.Hash.String()})
	if err := fs.tree.RebuildTreeWith(fs.leaves); err == nil {
		if err := fs.writeRoot(block.Number, fs.tree.MerkleRoot()); err != nil {
			return err
		}

		log.Debug("Add a new leaf", "number", block.Number, "root", hexutil.Encode(fs.tree.MerkleRoot())) //, "version", common.ToHex(version)) //MerkleRoot())

		return nil
	} else {
		return err
	}
}

func (fs *FileStorage) Root() common.Hash {
	return common.BytesToHash(fs.tree.MerkleRoot())
}

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
	log.Info("File DB Closed", "database", fs.db.Path())
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
		buk := tx.Bucket([]byte("blocks_" + fs.version))
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
		buk, err := tx.CreateBucketIfNotExists([]byte("files_" + fs.version))
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
	if b.Number < fs.LastListenBlockNumber {
		return nil
	}
	if record && b.Number > fs.CheckPoint {
		if err := fs.db.Update(func(tx *bolt.Tx) error {
			buk, err := tx.CreateBucketIfNotExists([]byte("blocks_" + fs.version))
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

			return buk.Put(k, v)
		}); err == nil {
			if err := fs.addLeaf(b); err == nil {
				if err := fs.writeCheckPoint(); err == nil {
					fs.CheckPoint = b.Number
				}
			}
		} else {
			return err
		}

	}

	fs.LastListenBlockNumber = b.Number
	return fs.writeBlockNumber()
}

func (fs *FileStorage) Version() string {
	return fs.version
}

func (fs *FileStorage) initBlocks() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte("blocks_" + fs.version)); err != nil {
			return err
		} else {
			c := buk.Cursor()

			for k, v := c.First(); k != nil; k, v = c.Next() {

				var x Block

				if err := json.Unmarshal(v, &x); err != nil {
					return err
				}
				fs.blocks = append(fs.blocks, &x)
			}
			sort.Slice(fs.blocks, func(i, j int) bool {
				return fs.blocks[i].Number < fs.blocks[j].Number
			})
			log.Info("Fs blocks initializing ... ...", "blocks", len(fs.blocks))
			return nil
		}
	})
}

func (fs *FileStorage) initFiles() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte("files_" + fs.version)); buk == nil || err != nil {
			return err
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

func (fs *FileStorage) readFsId() error {
	return fs.db.View(func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte("id_" + fs.version))
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
		fs.id = number

		return nil
	})
}

func (fs *FileStorage) ID() uint64 {
	return fs.id
}

func (fs *FileStorage) initFsId() error {
	err := fs.readFsId()
	if fs.id > 0 && err == nil {
		return nil
	}
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("id_" + fs.version))
		if err != nil {
			return err
		}
		//id := uint64(rand.Int63n(1 << 63 - 1))
		//id := uint64(rand.Int63n(1000))
		id := binary.LittleEndian.Uint64([]byte(uuid.NewRandom()))
		//id := uint64(os.Getuid())
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(id, 16)))
		fs.id = id //binary.LittleEndian.Uint64([]byte(id[:]))//uint64(id[:])

		return e
	})
}
func (fs *FileStorage) initCheckPoint() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("checkpoint_" + fs.version))
		if err != nil {
			return err
		}

		v := buk.Get([]byte("key"))

		if v == nil {
			//return ErrReadDataFromBoltDB
			return nil
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
	return fs.db.Update(func(tx *bolt.Tx) error {
		//buk := tx.Bucket([]byte("currentBlockNumber_" + fs.version))
		buk, err := tx.CreateBucketIfNotExists([]byte("currentBlockNumber_" + fs.version))
		if err != nil {
			return err
		}

		v := buk.Get([]byte("key"))

		if v == nil {
			//return ErrReadDataFromBoltDB
			return nil
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
		buk, err := tx.CreateBucketIfNotExists([]byte("checkpoint_" + fs.version))
		if err != nil {
			return err
		}
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.CheckPoint, 16)))

		return e
	})
}

func (fs *FileStorage) writeRoot(number uint64, root []byte) error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("version_" + fs.version))
		if err != nil {
			return err
		}
		e := buk.Put([]byte(strconv.FormatUint(number, 16)), root)

		return e
	})
}

func (fs *FileStorage) GetRootByNumber(number uint64) (root []byte) {
	cb := func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("version_" + fs.version))
		if err != nil {
			return err
		}

		v := buk.Get([]byte(strconv.FormatUint(number, 16)))

		if v == nil {
			return nil
		}

		root = v
		return nil
	}
	if err := fs.db.Update(cb); err != nil {
		return nil
	}

	return root
}

func (fs *FileStorage) writeBlockNumber() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("currentBlockNumber_" + fs.version))
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
