// Copyright 2023 The CortexTheseus Authors
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

package backend

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/merkletree"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	"strings"
	"sync"
	"sync/atomic"
	//lru "github.com/hashicorp/golang-lru"
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/google/uuid"
	bolt "go.etcd.io/bbolt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"
)

const (
	FILES_         = "files_"
	BLOCKS_        = "blocks_"
	ID_            = "id_"
	VERSION_       = "version_"
	TORRENT_       = "torrent_"
	CUR_BLOCK_NUM_ = "currentBlockNumber_"
)

type ChainDB struct {
	filesContractAddr map[common.Address]*types.FileInfo
	files             []*types.FileInfo //only storage init files from local storage
	blocks            []*types.Block    //only storage init ckp blocks from local storage
	txs               atomic.Uint64
	db                *bolt.DB
	version           string

	id                    atomic.Uint64
	checkPoint            atomic.Uint64
	lastListenBlockNumber atomic.Uint64
	leaves                []merkletree.Content
	tree                  *merkletree.MerkleTree
	dataDir               string
	config                *params.Config
	treeUpdates           time.Duration
	metrics               bool

	torrents map[string]uint64
	lock     sync.RWMutex
	//rootCache *lru.Cache

	initOnce sync.Once
}

func NewChainDB(config *params.Config) (*ChainDB, error) {
	if err := os.MkdirAll(config.DataDir, 0777); err != nil {
		log.Error("Make data dir failed", "err", err, "dir", config.DataDir)
		return nil, err
	}

	db, dbErr := bolt.Open(filepath.Join(config.DataDir,
		".file.bolt.db"), 0777, &bolt.Options{
		Timeout: time.Second,
	})
	if dbErr != nil {
		log.Error("Open database file failed", "err", dbErr, "dir", config.DataDir)
		return nil, dbErr
	}
	//db.NoSync = true

	fs := &ChainDB{
		filesContractAddr: make(map[common.Address]*types.FileInfo),
		db:                db,
		dataDir:           config.DataDir,
	}

	fs.config = config
	fs.metrics = config.Metrics
	fs.version = params.Version
	fs.torrents = make(map[string]uint64)

	//fs.rootCache, _ = lru.New(8)

	/*if err := fs.initBlockNumber(); err != nil {
		log.Error("Init block error", "err", err)
		return nil, err
	}
	//if err := fs.initCheckPoint(); err != nil {
	//	return nil, err
	//}
	//if err := fs.initBlocks(); err != nil {
	//	return nil, err
	//}
	if err := fs.initFiles(); err != nil {
		log.Error("Init files error", "err", err)
		return nil, err
	}
	if err := fs.initMerkleTree(); err != nil {
		log.Error("Init mkt error", "err", err)
		return nil, err
	}

	if err := fs.initID(); err != nil {
		log.Error("Init node id error", "err", err)
		return nil, err
	}*/

	//fs.history()

	log.Info("Storage ID generated", "id", fs.id.Load(), "version", fs.version)

	return fs, nil
}

func (fs *ChainDB) Init() (err error) {
	fs.initOnce.Do(func() {
		if err = fs.initBlockNumber(); err != nil {
			log.Error("Init block error", "err", err)
			//return err
		}

		if err = fs.initFiles(); err != nil {
			log.Error("Init files error", "err", err)
			//return err
		}
		if err = fs.initMerkleTree(); err != nil {
			log.Error("Init mkt error", "err", err)
			//return err
		}

		if err = fs.initID(); err != nil {
			log.Error("Init node id error", "err", err)
			//return err
		}
	})

	return
}

func (fs *ChainDB) Files() []*types.FileInfo {
	return fs.files
}

func (fs *ChainDB) Blocks() []*types.Block {
	return fs.blocks
}

func (fs *ChainDB) Torrents() map[string]uint64 {
	return fs.torrents
}

func (fs *ChainDB) Leaves() []merkletree.Content {
	return fs.leaves
}

func (fs *ChainDB) Txs() uint64 {
	return fs.txs.Load()
}

func (fs *ChainDB) Reset() error {
	fs.blocks = nil
	fs.checkPoint.Store(0)
	fs.lastListenBlockNumber.Store(0)
	if err := fs.initMerkleTree(); err != nil {
		return errors.New("err storage reset")
	}
	log.Warn("Storage status reset")
	return nil
}

func (fs *ChainDB) NewFileInfo(fileMeta *types.FileMeta) *types.FileInfo {
	ret := &types.FileInfo{Meta: fileMeta, LeftSize: fileMeta.RawSize}
	return ret
}

func (fs *ChainDB) initMerkleTree() error {
	if err := fs.initBlocks(); err != nil {
		return err
	}

	fs.leaves = nil
	fs.leaves = append(fs.leaves, merkletree.NewContent(params.MainnetGenesisHash.String(), uint64(0))) //BlockContent{X: params.MainnetGenesisHash.String()}) //"0x21d6ce908e2d1464bd74bbdbf7249845493cc1ba10460758169b978e187762c1"})
	tr, err := merkletree.NewTree(fs.leaves)
	if err != nil {
		return err
	}
	fs.tree = tr
	for _, block := range fs.blocks {
		if err := fs.addLeaf(block, false, false); err != nil {
			panic("Storage merkletree construct failed")
		}
	}

	log.Info("Storage merkletree initialization", "root", hexutil.Encode(fs.tree.MerkleRoot()), "number", fs.lastListenBlockNumber.Load(), "checkpoint", fs.checkPoint.Load(), "version", fs.version, "len", len(fs.blocks))

	return nil
}

func (fs *ChainDB) Metrics() time.Duration {
	return fs.treeUpdates
}

// Make sure the block group is increasing by number
func (fs *ChainDB) addLeaf(block *types.Block, mes bool, dup bool) error {
	if fs.tree == nil {
		return errors.New("mkt is nil")
	}

	number := block.Number
	leaf := merkletree.NewContent(block.Hash.String(), number)

	l, e := fs.tree.VerifyContent(leaf)
	if !l {
		if !dup {
			fs.leaves = append(fs.leaves, leaf)
		}
	} else {
		log.Debug("Node is already in the tree", "num", number, "len", len(fs.blocks), "leaf", len(fs.leaves), "ckp", fs.checkPoint.Load(), "mes", mes, "dup", dup, "err", e)
		if !mes {
			return nil
		}
	}

	if mes {
		log.Debug("Messing", "num", number, "len", len(fs.blocks), "leaf", len(fs.leaves), "ckp", fs.checkPoint.Load(), "mes", mes, "dup", dup)
		sort.Slice(fs.leaves, func(i, j int) bool {
			return fs.leaves[i].(merkletree.BlockContent).N() < fs.leaves[j].(merkletree.BlockContent).N()
		})

		i := sort.Search(len(fs.leaves), func(i int) bool { return fs.leaves[i].(merkletree.BlockContent).N() > number })

		if i > len(fs.leaves) {
			i = len(fs.leaves)
		}

		log.Warn("Messing solved", "num", number, "len", len(fs.blocks), "leaf", len(fs.leaves), "ckp", fs.checkPoint.Load(), "mes", mes, "dup", dup, "i", i)

		if err := fs.tree.RebuildTreeWith(fs.leaves[0:i]); err != nil {
			return err
		}

	} else {
		if err := fs.tree.AddNode(leaf); err != nil {
			return err
		}

		// TODO

		if number > fs.checkPoint.Load() {
			fs.checkPoint.Store(number)
		}
	}

	if err := fs.writeRoot(number, fs.tree.MerkleRoot()); err != nil {
		return err
	}
	return nil
}

func (fs *ChainDB) Root() common.Hash {
	if fs.tree == nil {
		return common.EmptyHash
	}
	return common.BytesToHash(fs.tree.MerkleRoot())
}

func (fs *ChainDB) AddFile(x *types.FileInfo) (uint64, bool, error) {
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

func (fs *ChainDB) GetFileByAddr(addr common.Address) *types.FileInfo {
	if f, ok := fs.filesContractAddr[addr]; ok {
		return f
	}
	return nil
}

func (fs *ChainDB) Close() error {
	defer fs.db.Close()
	log.Info("File DB Closed", "database", fs.db.Path(), "last", fs.lastListenBlockNumber.Load())
	return fs.Flush()
}

var (
	ErrReadDataFromBoltDB = errors.New("bolt DB Read Error")
)

func uint64ToBytes(i uint64) []byte {
	var buf [8]byte
	binary.BigEndian.PutUint64(buf[:], i)
	return buf[:]
}

func (fs *ChainDB) GetBlockByNumber(blockNum uint64) *types.Block {
	var block types.Block

	cb := func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte(BLOCKS_ + fs.version))
		if buk == nil {
			return ErrReadDataFromBoltDB
		}
		k := uint64ToBytes(blockNum)

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

func (fs *ChainDB) progress(f *types.FileInfo, init bool) (bool, error) {
	update := false
	err := fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(FILES_ + fs.version))
		if err != nil {
			log.Error("Progress bucket failed", "err", err)
			return err
		}

		k := []byte(f.Meta.InfoHash)
		var v []byte
		bef := buk.Get(k)
		if bef == nil {
			update = true
			v, err = json.Marshal(f)
			if err != nil {
				log.Error("Progress json failed", "err", err)
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
						log.Debug("New relate file found and progressing", "hash", info.Meta.InfoHash, "old", info.ContractAddr, "new", f.ContractAddr, "relate", len(info.Relate), "init", init)
						f.Relate = append(f.Relate, *info.ContractAddr)
					} else {
						log.Debug("Address changed and progressing", "hash", info.Meta.InfoHash, "old", info.ContractAddr, "new", f.ContractAddr, "relate", len(info.Relate), "init", init)
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
					log.Debug("New relate file found", "hash", info.Meta.InfoHash, "old", info.ContractAddr, "new", f.ContractAddr, "r", len(info.Relate), "l", info.LeftSize, "r", len(f.Relate), "l", f.LeftSize, "init", init)
					f.Relate = info.Relate
					return buk.Put(k, v)
				}
			}
		}
		return nil
	})

	return update, err
}

// func (fs *ChainDB) addBlock(b *Block, record bool) error {
func (fs *ChainDB) AddBlock(b *types.Block) error {
	if fs.metrics {
		defer func(start time.Time) { fs.treeUpdates += time.Since(start) }(time.Now())
	}
	//i := sort.Search(len(fs.blocks), func(i int) bool { return fs.blocks[i].Number > b.Number })
	//if i == len(fs.blocks) {
	//todo
	//} else {
	//	log.Warn("Encounter ancient block (dup)", "cur", b.Number, "index", i, "len", len(fs.blocks), "ckp", fs.CheckPoint)
	//	return nil
	//}
	ancient := fs.GetBlockByNumber(b.Number)
	if ancient != nil && ancient.Hash == b.Hash {
		fs.addLeaf(b, false, true)
		return nil
	}

	if err := fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(BLOCKS_ + fs.version))
		if err != nil {
			return err
		}
		v, err := json.Marshal(b)
		if err != nil {
			return err
		}
		k := uint64ToBytes(b.Number)

		return buk.Put(k, v)
	}); err == nil {
		fs.blocks = append(fs.blocks, b)
		fs.txs.Add(uint64(len(b.Txs)))
		mes := false
		if b.Number < fs.checkPoint.Load() {
			mes = true
		}

		fs.addLeaf(b, mes, false)
	} else {
		return err
	}
	if b.Number > fs.lastListenBlockNumber.Load() {
		fs.lastListenBlockNumber.Store(b.Number)
		if err := fs.Flush(); err != nil {
			return err
		}
	}
	return nil
}

func (fs *ChainDB) Version() string {
	return fs.version
}

func (fs *ChainDB) initBlocks() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte(BLOCKS_ + fs.version)); err != nil {
			return err
		} else {
			c := buk.Cursor()

			for k, v := c.First(); k != nil; k, v = c.Next() {

				var x types.Block

				if err := json.Unmarshal(v, &x); err != nil {
					return err
				}
				fs.blocks = append(fs.blocks, &x)
				fs.txs.Add(uint64(len(x.Txs)))
			}
			sort.Slice(fs.blocks, func(i, j int) bool {
				return fs.blocks[i].Number < fs.blocks[j].Number
			})
			log.Info("Fs blocks initializing ... ...", "blocks", len(fs.blocks), "txs", fs.txs.Load())
			return nil
		}
	})
}

func (fs *ChainDB) history() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte(VERSION_ + fs.version)); err != nil {
			return err
		} else {
			c := buk.Cursor()

			for k, v := c.First(); k != nil; k, v = c.Next() {
				log.Info("History", "k", string(k), "v", common.BytesToHash(v))
			}
			return nil
		}
	})
}

func (fs *ChainDB) initFiles() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte(FILES_ + fs.version)); buk == nil || err != nil {
			return err
		} else {
			c := buk.Cursor()

			for k, v := c.First(); k != nil; k, v = c.Next() {

				var x types.FileInfo
				if err := json.Unmarshal(v, &x); err != nil {
					log.Error("Json unmarshal error", "err", err)
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

func (fs *ChainDB) ID() uint64 {
	return fs.id.Load()
}

func (fs *ChainDB) initID() error {
	if err := fs.db.View(func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte(ID_ + fs.version))
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
		fs.id.Store(number)

		return nil
	}); fs.id.Load() > 0 && err == nil {
		return nil
	}

	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(ID_ + fs.version))
		if err != nil {
			return err
		}
		uid, err := uuid.NewRandom()
		if err != nil {
			return err
		}
		id := binary.LittleEndian.Uint64([]byte(uid[:]))
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(id, 16)))
		fs.id.Store(id) //binary.LittleEndian.Uint64([]byte(id[:]))//uint64(id[:])

		return e
	})
}

/*func (fs *ChainDB) initCheckPoint() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("checkpoint_" + fs.version))
		if err != nil {
			return err
		}

		v := buk.Get([]byte("key"))

		if v == nil {
			log.Warn("Start from check point (default:0)")
			return nil
		}

		number, err := strconv.ParseUint(string(v), 16, 64)
		if err != nil {
			return err
		}

		fs.CheckPoint = number
		log.Info("Start from check point (default:0)", "num", number)

		return nil
	})
}*/

func (fs *ChainDB) initBlockNumber() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(CUR_BLOCK_NUM_ + fs.version))
		if err != nil {
			return err
		}

		v := buk.Get([]byte("key"))

		if v == nil {
			log.Warn("Start from block number (default:0)")
			return nil
		}

		number, err := strconv.ParseUint(string(v), 16, 64)
		if err != nil {
			return err
		}

		fs.lastListenBlockNumber.Store(number)
		log.Info("Start from block number (default:0)", "num", number)

		return nil
	})
}

/*func (fs *ChainDB) writeCheckPoint() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("checkpoint_" + fs.version))
		if err != nil {
			return err
		}
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.CheckPoint, 16)))

		return e
	})
}*/

func (fs *ChainDB) writeRoot(number uint64, root []byte) error {
	//fs.rootCache.Add(number, root)
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(VERSION_ + fs.version))
		if err != nil {
			return err
		}
		e := buk.Put([]byte(strconv.FormatUint(number, 16)), root)

		if e == nil {
			log.Debug("Root update", "number", number, "root", common.BytesToHash(root))
		}

		return e
	})
}

func (fs *ChainDB) GetRoot(number uint64) (root []byte) {
	//if root, suc := fs.rootCache.Get(number); suc {
	//	return root.([]byte)
	//}
	cb := func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte(VERSION_ + fs.version))
		if buk == nil {
			return errors.New("root bucket not exist")
		}

		v := buk.Get([]byte(strconv.FormatUint(number, 16)))
		if v == nil {
			return errors.New("root value not exist")
		}

		root = v
		return nil
	}
	if err := fs.db.View(cb); err != nil {
		return nil
	}

	return root
}

func (fs *ChainDB) Flush() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(CUR_BLOCK_NUM_ + fs.version))
		if err != nil {
			return err
		}
		log.Trace("Write block number", "num", fs.lastListenBlockNumber.Load())
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.lastListenBlockNumber.Load(), 16)))

		return e
	})
}

func (fs *ChainDB) SkipPrint() {
	var (
		str  string
		from uint64
	)
	for _, b := range fs.blocks {
		//		if b.Number < 395964 {
		//			continue
		//		}
		//Skip{From: 160264, To: 395088},
		if b.Number-from > 1000 {
			str = str + "{From:" + strconv.FormatUint(from, 10) + ",To:" + strconv.FormatUint(b.Number, 10) + "},"
		}
		from = b.Number
		//fmt.Println(b.Number, ":true,")
	}

	if fs.lastListenBlockNumber.Load()-from > 1000 {
		str = str + "{From:" + strconv.FormatUint(from, 10) + ",To:" + strconv.FormatUint(fs.lastListenBlockNumber.Load(), 10) + "},"
	}

	//log.Info("Skip chart", "skips", str)
	fmt.Println(str)
}

// SetTorrent is for recording torrent latest status
func (fs *ChainDB) SetTorrentProgress(ih string, size uint64) (bool, uint64, error) {
	fs.lock.Lock()
	defer fs.lock.Unlock()

	if s, ok := fs.torrents[ih]; ok {
		if s >= size {
			return false, s, nil
		}
	}
	if err := fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(TORRENT_ + fs.version))
		if err != nil {
			return err
		}
		v := buk.Get([]byte(ih))

		if v == nil {
			err = buk.Put([]byte(ih), []byte(strconv.FormatUint(size, 16)))
		} else {
			s, err := strconv.ParseUint(string(v), 16, 64)
			if err != nil {
				return err
			}
			if size > s {
				err = buk.Put([]byte(ih), []byte(strconv.FormatUint(size, 16)))
			} else {
				size = s
			}
		}

		return err
	}); err != nil {
		return false, 0, err
	}

	fs.torrents[ih] = size

	log.Debug("File status has been changed", "ih", ih, "size", size, "count", len(fs.torrents))

	return true, size, nil
}

// GetTorrent return the torrent status by uint64, if return 0 for torrent not exist
func (fs *ChainDB) GetTorrentProgress(ih string) (progress uint64, err error) {
	fs.lock.RLock()
	defer fs.lock.RUnlock()

	//TODO
	ih = strings.ToLower(ih)

	if s, ok := fs.torrents[ih]; ok {
		return s, nil
	}
	cb := func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte(TORRENT_ + fs.version))
		if buk == nil {
			return errors.New("root bucket not exist")
		}

		v := buk.Get([]byte(ih))

		if v == nil {
			return errors.New("No torrent record found")
		}

		s, err := strconv.ParseUint(string(v), 16, 64)
		if err != nil {
			return err
		}

		progress = s

		return nil
	}
	if err := fs.db.View(cb); err != nil {
		return 0, err
	}

	return progress, nil
}

func (fs *ChainDB) InitTorrents() (map[string]uint64, error) {
	err := fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte(TORRENT_ + fs.version)); err != nil {
			return err
		} else {
			c := buk.Cursor()
			for k, v := c.First(); k != nil; k, v = c.Next() {
				size, err := strconv.ParseUint(string(v), 16, 64)
				if err != nil {
					return err
				}
				fs.torrents[string(k)] = size
			}
			log.Debug("Torrent initializing ... ...", "torrents", len(fs.torrents))
			return nil
		}
	})
	if err != nil {
		return nil, err
	}
	return fs.torrents, nil
}

func (fs *ChainDB) CheckPoint() uint64 {
	return fs.checkPoint.Load()
}

func (fs *ChainDB) LastListenBlockNumber() uint64 {
	return fs.lastListenBlockNumber.Load()
}

func (fs *ChainDB) Anchor(n uint64) {
	fs.lastListenBlockNumber.Store(n)
}
