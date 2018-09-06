package types

import (
	"bytes"
	"errors"
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
}

// NewFileStorage ...
func NewFileStorage() *FileStorage {
	return &FileStorage{
		make(map[metainfo.Hash]*FileInfo),
		make(map[common.Address]*FileInfo),
		make(map[uint64]bool),
		make(map[uint64]*Block),
		0,
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
	return nil
}

// HasBlock ...
func (fs *FileStorage) HasBlock(blockNum uint64) bool {
	if _, ok := fs.blockMap[blockNum]; ok {
		return true
	}
	return false
}

// GetBlock ...
func (fs *FileStorage) GetBlock(blockNum uint64) *Block {
	b, _ := fs.blockMap[blockNum]
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

type boltDBTorrent struct {
	cl *boltDBClient
	ih metainfo.Hash
}

// NewBoltDB ...
func NewBoltDB(filePath string) *boltDBClient {
	db, err := bolt.Open(filepath.Join(filePath, "bolt.db"), 0600, &bolt.Options{
		Timeout: time.Second,
	})
	expect.Nil(err)
	db.NoSync = true
	return &boltDBClient{db}
}

func (me *boltDBClient) Close() error {
	return me.db.Close()
}

func (me *boltDBClient) OpenTorrent(info *metainfo.Info, infoHash metainfo.Hash) (*boltDBTorrent, error) {
	return &boltDBTorrent{me, infoHash}, nil
}

func (boltDBTorrent) Close() error { return nil }
