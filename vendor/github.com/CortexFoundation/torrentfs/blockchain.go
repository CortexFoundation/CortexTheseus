// Copyright 2020 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.
package torrentfs

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	"github.com/pborman/uuid"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"crypto/sha256"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	bolt "go.etcd.io/bbolt"
)

type ChainIndex struct {
	filesContractAddr map[common.Address]*types.FileInfo
	files             []*types.FileInfo //only storage init files from local storage
	blocks            []*types.Block    //only storage init ckp blocks from local storage
	txs               uint64
	db                *bolt.DB
	version           string

	id                    uint64
	CheckPoint            uint64
	LastListenBlockNumber uint64
	leaves                []types.Content
	tree                  *types.MerkleTree
	dataDir               string
	config                *Config
	treeUpdates           time.Duration
	metrics               bool
}

func NewChainIndex(config *Config) (*ChainIndex, error) {

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

	fs := &ChainIndex{
		filesContractAddr: make(map[common.Address]*types.FileInfo),
		db:                db,
		dataDir:           config.DataDir,
	}

	fs.config = config

	fs.metrics = config.Metrics

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

	if err := fs.initFsId(); err != nil {
		return nil, err
	}

	log.Info("Storage ID generated", "id", fs.id)

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
func (t BlockContent) Equals(other types.Content) (bool, error) {
	return t.x == other.(BlockContent).x, nil
}

func (fs *ChainIndex) Files() []*types.FileInfo {
	return fs.files
}

func (fs *ChainIndex) Blocks() []*types.Block {
	return fs.blocks
}

func (fs *ChainIndex) Txs() uint64 {
	return fs.txs
}

func (fs *ChainIndex) Reset() error {
	fs.blocks = nil
	fs.CheckPoint = 0
	fs.LastListenBlockNumber = 0
	if err := fs.initMerkleTree(); err != nil {
		return errors.New("err storage reset")
	}
	log.Warn("Storage status reset")
	return nil
}

func (fs *ChainIndex) NewFileInfo(Meta *types.FileMeta) *types.FileInfo {
	ret := &types.FileInfo{Meta, nil, Meta.RawSize, nil}
	return ret
}

func (fs *ChainIndex) initMerkleTree() error {
	fs.leaves = nil
	fs.leaves = append(fs.leaves, BlockContent{x: params.MainnetGenesisHash.String()}) //"0x21d6ce908e2d1464bd74bbdbf7249845493cc1ba10460758169b978e187762c1"})
	tr, err := types.NewTree(fs.leaves)
	if err != nil {
		return err
	}
	fs.tree = tr
	for _, block := range fs.blocks {
		if err := fs.addLeaf(block, true); err != nil {
			panic("Storage merkletree construct failed")
		}
	}

	log.Info("Storage merkletree initialization", "root", hexutil.Encode(fs.tree.MerkleRoot()), "number", fs.LastListenBlockNumber, "checkpoint", fs.CheckPoint, "version", fs.version)

	return nil
}

func (fs *ChainIndex) Metrics() time.Duration {
	return fs.treeUpdates
}

//Make sure the block group is increasing by number
func (fs *ChainIndex) addLeaf(block *types.Block, init bool) error {
	number := block.Number
	leaf := BlockContent{x: block.Hash.String()}

	if len(fs.leaves) >= params.LEAFS {
		fs.leaves = nil
		fs.leaves = append(fs.leaves, BlockContent{x: hexutil.Encode(fs.tree.MerkleRoot())})
		log.Debug("Next tree level", "leaf", len(fs.leaves), "root", hexutil.Encode(fs.tree.MerkleRoot()))
	}

	fs.leaves = append(fs.leaves, leaf)

	if err := fs.tree.RebuildTreeWith(fs.leaves); err == nil {
		if err := fs.writeRoot(number, fs.tree.MerkleRoot()); err != nil {
			return err
		}

		return nil
	} else {
		return err
	}
}

func (fs *ChainIndex) Root() common.Hash {
	return common.BytesToHash(fs.tree.MerkleRoot())
}

func (fs *ChainIndex) UpdateFile(x *types.FileInfo) (uint64, bool, error) {
	if fs.metrics {
		defer func(start time.Time) { fs.treeUpdates += time.Since(start) }(time.Now())
	}

	addr := *x.ContractAddr
	if _, ok := fs.filesContractAddr[addr]; ok {
		update, err := fs.progress(x, false)
		if err != nil {
			return 0, update, err
		}

		fs.filesContractAddr[addr] = x
		return 0, update, nil
	}

	update, err := fs.progress(x, true)
	if err != nil {
		return 0, update, err
	}

	fs.filesContractAddr[addr] = x

	if !update {
		return 0, update, nil
	}

	fs.files = append(fs.files, x)

	return 1, update, nil
}

func (fs *ChainIndex) GetFileByAddr(addr common.Address) *types.FileInfo {
	if f, ok := fs.filesContractAddr[addr]; ok {
		return f
	}
	return nil
}

func (fs *ChainIndex) Close() error {
	defer fs.db.Close()
	fs.writeCheckPoint()
	log.Info("File DB Closed", "database", fs.db.Path())
	return fs.Flush()
}

var (
	ErrReadDataFromBoltDB = errors.New("bolt DB Read Error")
)

func (fs *ChainIndex) GetBlockByNumber(blockNum uint64) *types.Block {
	var block types.Block

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

func (fs *ChainIndex) progress(f *types.FileInfo, init bool) (bool, error) {
	update := false
	err := fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("files_" + fs.version))
		if err != nil {
			return err
		}

		k, err := json.Marshal(f.Meta.InfoHash)
		if err != nil {
			return err
		}
		var v []byte
		bef := buk.Get(k)
		if bef == nil {
			update = true
			v, err = json.Marshal(f)
			if err != nil {
				return err
			}
			return buk.Put(k, v)
		} else {
			var info types.FileInfo
			if err := json.Unmarshal(bef, &info); err != nil {
				update = true
				return buk.Put(k, v)
			}

			if info.LeftSize > f.LeftSize {
				update = true
				if *info.ContractAddr != *f.ContractAddr {
					var insert = true
					for _, addr := range info.Relate {
						if *f.ContractAddr == addr {
							insert = false
							break
						}
					}
					if insert {
						log.Debug("New relate file found and progressing", "hash", info.Meta.InfoHash.String(), "old", info.ContractAddr, "new", f.ContractAddr, "relate", len(info.Relate), "init", init)
						f.Relate = append(f.Relate, *info.ContractAddr)
					} else {
						log.Debug("Address changed and progressing", "hash", info.Meta.InfoHash.String(), "old", info.ContractAddr, "new", f.ContractAddr, "relate", len(info.Relate), "init", init)
					}
				}
				v, err = json.Marshal(f)
				if err != nil {
					return err
				}
				return buk.Put(k, v)
			} else {
				if *info.ContractAddr != *f.ContractAddr {
					for _, addr := range info.Relate {
						if *f.ContractAddr == addr {
							return nil
						}
					}
					info.Relate = append(info.Relate, *f.ContractAddr)
					v, err = json.Marshal(info)
					if err != nil {
						return err
					}
					log.Debug("New relate file found", "hash", info.Meta.InfoHash.String(), "old", info.ContractAddr, "new", f.ContractAddr, "r", len(info.Relate), "l", info.LeftSize, "r", len(f.Relate), "l", f.LeftSize, "init", init)
					f.Relate = info.Relate
					return buk.Put(k, v)
				}
			}
		}
		return nil
	})

	return update, err
}

//func (fs *ChainIndex) addBlock(b *Block, record bool) error {
func (fs *ChainIndex) AddBlock(b *types.Block) error {
	if b.Number < fs.LastListenBlockNumber {
		return nil
	}

	if fs.metrics {
		defer func(start time.Time) { fs.treeUpdates += time.Since(start) }(time.Now())
	}
	if b.Number > fs.CheckPoint {

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
			if err := fs.appendBlock(b); err == nil {
				fs.txs += uint64(len(b.Txs))
				if err := fs.addLeaf(b, false); err == nil {
					if err := fs.writeCheckPoint(); err == nil {
						fs.CheckPoint = b.Number
					}
				}
			}
		} else {
			return err
		}
	}

	fs.LastListenBlockNumber = b.Number
	return fs.Flush()
}

func (fs *ChainIndex) appendBlock(b *types.Block) error {
	if len(fs.blocks) == 0 || fs.blocks[len(fs.blocks)-1].Number < b.Number {
		log.Debug("Append block", "number", b.Number)
		fs.blocks = append(fs.blocks, b)
	} else {
		return errors.New("err block duplicated")
	}
	return nil
}

func (fs *ChainIndex) Version() string {
	return fs.version
}

func (fs *ChainIndex) initBlocks() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte("blocks_" + fs.version)); err != nil {
			return err
		} else {
			c := buk.Cursor()

			for k, v := c.First(); k != nil; k, v = c.Next() {

				var x types.Block

				if err := json.Unmarshal(v, &x); err != nil {
					return err
				}
				fs.blocks = append(fs.blocks, &x)
				fs.txs += uint64(len(x.Txs))
			}
			sort.Slice(fs.blocks, func(i, j int) bool {
				return fs.blocks[i].Number < fs.blocks[j].Number
			})
			log.Info("Fs blocks initializing ... ...", "blocks", len(fs.blocks), "txs", fs.txs)
			return nil
		}
	})
}

func (fs *ChainIndex) initFiles() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte("files_" + fs.version)); buk == nil || err != nil {
			return err
		} else {
			c := buk.Cursor()

			for k, v := c.First(); k != nil; k, v = c.Next() {

				var x types.FileInfo

				if err := json.Unmarshal(v, &x); err != nil {
					return err
				}
				fs.filesContractAddr[*x.ContractAddr] = &x
				fs.files = append(fs.files, &x)
				if x.Relate == nil {
					x.Relate = append(x.Relate, *x.ContractAddr)
				}
				for _, addr := range x.Relate {
					if _, ok := fs.filesContractAddr[addr]; !ok {
						tmp := x
						tmp.ContractAddr = &addr
						fs.filesContractAddr[addr] = &tmp
					}
				}
			}
			log.Info("File init finished", "files", len(fs.files), "total", len(fs.filesContractAddr))
			return nil
		}
	})
}

func (fs *ChainIndex) readFsId() error {
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

func (fs *ChainIndex) ID() uint64 {
	return fs.id
}

func (fs *ChainIndex) initFsId() error {
	err := fs.readFsId()
	if fs.id > 0 && err == nil {
		return nil
	}
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("id_" + fs.version))
		if err != nil {
			return err
		}
		id := binary.LittleEndian.Uint64([]byte(uuid.NewRandom()))
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(id, 16)))
		fs.id = id //binary.LittleEndian.Uint64([]byte(id[:]))//uint64(id[:])

		return e
	})
}
func (fs *ChainIndex) initCheckPoint() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("checkpoint_" + fs.version))
		if err != nil {
			return err
		}

		v := buk.Get([]byte("key"))

		if v == nil {
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

func (fs *ChainIndex) initBlockNumber() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("currentBlockNumber_" + fs.version))
		if err != nil {
			return err
		}

		v := buk.Get([]byte("key"))

		if v == nil {
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

func (fs *ChainIndex) writeCheckPoint() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("checkpoint_" + fs.version))
		if err != nil {
			return err
		}
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.CheckPoint, 16)))

		return e
	})
}

func (fs *ChainIndex) writeRoot(number uint64, root []byte) error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("version_" + fs.version))
		if err != nil {
			return err
		}
		e := buk.Put([]byte(strconv.FormatUint(number, 16)), root)

		return e
	})
}

func (fs *ChainIndex) GetRootByNumber(number uint64) (root []byte) {
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

func (fs *ChainIndex) Flush() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("currentBlockNumber_" + fs.version))
		if err != nil {
			return err
		}
		log.Trace("Write block number", "num", fs.LastListenBlockNumber)
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.LastListenBlockNumber, 16)))

		return e
	})
}
