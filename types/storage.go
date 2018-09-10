package types

import (
	"bytes"
	"encoding/json"
	"errors"
	"log"
	"path/filepath"
	"time"

	"github.com/anacrolix/missinggo/expect"
	"github.com/boltdb/bolt"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/anacrolix/torrent/metainfo"
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
	filesInfoHash     map[metainfo.Hash]*FileInfo
	filesContractAddr map[common.Address]*FileInfo
	blockChecked      map[uint64]bool
	blockMap          map[uint64]*Block
	LatestBlockNumber uint64
	db                *boltDBClient
}

// NewFileStorage ...
func NewFileStorage(flag *Flag) *FileStorage {
	db := NewBoltDB(*flag.DataDir)
	return &FileStorage{
		make(map[metainfo.Hash]*FileInfo),
		make(map[common.Address]*FileInfo),
		make(map[uint64]bool),
		make(map[uint64]*Block),
		0,
		db,
	}
}

// AddFile ...
func (fs *FileStorage) AddFile(x *FileInfo) error {
	ih := *x.Meta.InfoHash()
	if _, ok := fs.filesInfoHash[ih]; ok {
		return errors.New("file already existed")
	}
	addr := *x.ContractAddr
	if _, ok := fs.filesContractAddr[addr]; ok {
		return errors.New("file already existed")
	}
	fs.filesInfoHash[ih] = x
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

// GetFileByInfoHash ...
func (fs *FileStorage) GetFileByInfoHash(ih metainfo.Hash) *FileInfo {
	if f, ok := fs.filesInfoHash[ih]; ok {
		return f
	}
	return nil
}

// AddBlock ...
func (fs *FileStorage) AddBlock(b *Block) error {
	if _, ok := fs.blockMap[b.Number]; ok {
		return errors.New("block already existed")
	}
	if b.Number > fs.LatestBlockNumber {
		fs.LatestBlockNumber = b.Number
	}
	if b.Number > 0 {
		pb := fs.GetBlock(b.Number - 1)
		if pb != nil && !bytes.Equal(pb.Hash.Bytes(), b.ParentHash.Bytes()) {
			return errors.New("verify block hash failed")
		}
	}
	nb := fs.GetBlock(b.Number + 1)
	if nb != nil && !bytes.Equal(nb.ParentHash.Bytes(), b.Hash.Bytes()) {
		return errors.New("verify block hash failed")
	}
	fs.blockMap[b.Number] = b
	if t, err := fs.db.OpenBlock(b); err == nil {
		t.Write()
		t.Close()
	}
	return nil
}

// HasBlock ...
func (fs *FileStorage) HasBlock(blockNum uint64) bool {
	if _, ok := fs.blockMap[blockNum]; ok {
		return true
	} else if fs.GetBlock(blockNum) != nil {
		return true
	}
	return false
}

// GetBlock ...
func (fs *FileStorage) GetBlock(blockNum uint64) *Block {
	b, ok := fs.blockMap[blockNum]
	if !ok {
		t := fs.db.GetBlock(blockNum)
		if t != nil {
			fs.blockMap[blockNum] = t
			return t
		}
	}
	return b
}

// SetBlockChecked ...
func (fs *FileStorage) SetBlockChecked(blockNum uint64) error {
	if _, ok := fs.blockChecked[blockNum]; ok {
		return errors.New("block was already checked")
	}
	fs.blockChecked[blockNum] = true
	return nil
}

// IsBlockChecked ...
func (fs *FileStorage) IsBlockChecked(blockNum uint64) bool {
	if _, ok := fs.blockChecked[blockNum]; ok {
		return true
	}
	return false
}

// FlowControlMeta ...
type FlowControlMeta struct {
	InfoHash       metainfo.Hash
	BytesRequested uint64
}

type boltDBClient struct {
	db *bolt.DB
}

type boltDBBlock struct {
	cl *boltDBClient
	b  *Block
}

// NewBoltDB ...
func NewBoltDB(filePath string) *boltDBClient {
	db, err := bolt.Open(filepath.Join(filePath, ".file.bolt.db"), 0600, &bolt.Options{
		Timeout: time.Second,
	})
	expect.Nil(err)
	db.NoSync = true
	return &boltDBClient{db}
}

func (me *boltDBClient) Close() error {
	return me.db.Close()
}

func (me *boltDBClient) OpenBlock(b *Block) (*boltDBBlock, error) {
	return &boltDBBlock{me, b}, nil
}

func (me *boltDBClient) GetBlock(blockNum uint64) *Block {
	tx, err := me.db.Begin(false)
	if err != nil {
		return nil
	}
	b := tx.Bucket([]byte("blocks"))
	if b == nil {
		return nil
	}
	k, err := json.Marshal(blockNum)
	if err != nil {
		return nil
	}
	v := b.Get(k)
	if v == nil || len(v) == 0 {
		return nil
	}
	var block Block
	log.Println(blockNum, v)
	json.Unmarshal(v, &block)
	return &block
}

func (f *boltDBBlock) Write() error {
	f.cl.db.Update(func(tx *bolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte("blocks"))
		if err != nil {
			return err
		}
		v, err := json.Marshal(f.b)
		if err != nil {
			return err
		}
		k, err := json.Marshal(f.b.Number)
		if err != nil {
			return err
		}
		return b.Put(k, v)
	})
	return nil
}

func (boltDBBlock) Close() error { return nil }
