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
	"github.com/CortexFoundation/merkletree"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	"sync"
	"sync/atomic"
	//lru "github.com/hashicorp/golang-lru"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/google/uuid"
	bolt "go.etcd.io/bbolt"
	"os"
	"path/filepath"
	"strconv"
	"time"
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

	return fs, nil
}

func (fs *ChainDB) Init() (err error) {
	fs.initOnce.Do(func() {
		if err = fs.InitBlockNumber(); err != nil {
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

		log.Info("Storage ID generated", "id", fs.id.Load(), "version", fs.version)
	})

	return
}

/*func (fs *ChainDB) Reset() error {
	fs.blocks = nil
	fs.checkPoint.Store(0)
	fs.lastListenBlockNumber.Store(0)
	if err := fs.initMerkleTree(); err != nil {
		return errors.New("err storage reset")
	}
	log.Warn("Storage status reset")
	return nil
}*/

func (fs *ChainDB) Close() error {
	defer fs.db.Close()
	log.Info("File DB Closed", "database", fs.db.Path(), "last", fs.lastListenBlockNumber.Load())
	return fs.Flush()
}

func (fs *ChainDB) Version() string {
	return fs.version
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

		log.Info("New random id generated !!!", "key", ID_+fs.version, "id", id)
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
}

func (fs *ChainDB) writeCheckPoint() error {
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte("checkpoint_" + fs.version))
		if err != nil {
			return err
		}
		e := buk.Put([]byte("key"), []byte(strconv.FormatUint(fs.CheckPoint, 16)))

		return e
	})
}*/

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
