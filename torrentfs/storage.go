package torrentfs

import (
	"bytes"
	"encoding/json"
	"errors"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/boltdb/bolt"

	"github.com/anacrolix/torrent/metainfo"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/common/hexutil"
	"github.com/ethereum/go-ethereum/log"
)

const (
	// Chosen to match the usual chunk size in a torrent client. This way,
	// most chunk writes are to exactly one full item in bolt DB.
	chunkSize = 1 << 14
)

// FileInfo ...
type FileInfo struct {
	Meta *FileMeta
	// Transaction hash
	TxHash *common.Hash
	// Contract Address
	ContractAddr *common.Address
	LeftSize     uint64
}

// NewFileInfo ...
func NewFileInfo(Meta *FileMeta) *FileInfo {
	return &FileInfo{Meta, nil, nil, Meta.RawSize}
}

// FileStorage ...
type FileStorage struct {
	// filesInfoHash     map[metainfo.Hash]*FileInfo
	filesContractAddr map[common.Address]*FileInfo
	blockChecked      map[uint64]bool
	// db                *boltDBClient
	db *bolt.DB

	LatestBlockNumber atomic.Value
	lock              sync.RWMutex
}

// NewFileStorage ...
func NewFileStorage(config *Config) (*FileStorage, error) {
	db, dbErr := bolt.Open(filepath.Join(config.DataDir,
		".file.bolt.db"), 0600, &bolt.Options{
		Timeout: time.Second,
	})
	if dbErr != nil {
		return nil, dbErr
	}
	db.NoSync = true

	// db := NewBoltDB(config.DataDir)
	return &FileStorage{
		// filesInfoHash:     make(map[metainfo.Hash]*FileInfo),
		filesContractAddr: make(map[common.Address]*FileInfo),
		blockChecked:      make(map[uint64]bool),
		db:                db,
	}, nil
}

func (fs *FileStorage) CurrentBlockNumber() *hexutil.Uint64 {
	return fs.LatestBlockNumber.Load().(*hexutil.Uint64)
}

// AddFile ...
func (fs *FileStorage) AddFile(x *FileInfo) error {
	// ih := *x.Meta.InfoHash()
	// if _, ok := fs.filesInfoHash[ih]; ok {
	// 	return errors.New("file already existed")
	// }
	addr := *x.ContractAddr
	if _, ok := fs.filesContractAddr[addr]; ok {
		return errors.New("file already existed")
	}
	// fs.filesInfoHash[ih] = x
	fs.filesContractAddr[addr] = x
	return nil
}

// GetFileByAddr ...
func (fs *FileStorage) GetFileByAddr(addr common.Address) *FileInfo {
	if f, ok := fs.filesContractAddr[addr]; ok {
		return f
	}
	return nil
}

// func (fs *FileStorage) GetFileByInfoHash(ih metainfo.Hash) *FileInfo {
// 	if f, ok := fs.filesInfoHash[ih]; ok {
// 		return f
// 	}
// 	return nil
// }

// AddBlock ...
func (fs *FileStorage) AddBlock(b *Block) error {
	if fs.HasBlock(b.Number) {
		return errors.New("block already existed")
	}

	if b.Number > 0 {
		pb := fs.GetBlockByNumber(b.Number - 1)
		if pb != nil && !bytes.Equal(pb.Hash.Bytes(), b.ParentHash.Bytes()) {
			return errors.New("verify block hash failed")
		}
	}
	nb := fs.GetBlockByNumber(b.Number + 1)
	if nb != nil && !bytes.Equal(nb.ParentHash.Bytes(), b.Hash.Bytes()) {
		return errors.New("verify block hash failed")
	}

	err := fs.WriteBlock(b)
	return err
}

// HasBlock ...
func (fs *FileStorage) HasBlock(blockNum uint64) bool {
	if b := fs.GetBlockByNumber(blockNum); b != nil {
		return true
	}
	return false
}

func (fs *FileStorage) Close() error {
	log.Info("Torrent File Storage Closed", "database dir", fs.db.Path())
	return fs.db.Close()
}

var (
	ErrReadDataFromBoltDB = errors.New("Bolt DB Read Error")
)

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

		fs.lock.RLock()
		v := buk.Get(k)
		fs.lock.RUnlock()

		if v == nil || len(v) == 0 {
			return ErrReadDataFromBoltDB
		}
		//	log.Println(blockNum, v)
		json.Unmarshal(v, &block)
		return nil
	}

	if err := fs.db.View(cb); err != nil {
		return nil
	}
	return &block
}

func (fs *FileStorage) WriteBlock(b *Block) error {
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

	return err
}

// FlowControlMeta ...
type FlowControlMeta struct {
	InfoHash       metainfo.Hash
	BytesRequested uint64
}
