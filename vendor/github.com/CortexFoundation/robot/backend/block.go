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
	"encoding/json"
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/types"
	bolt "go.etcd.io/bbolt"
	"sort"
	"strconv"
	"time"
)

func (fs *ChainDB) Blocks() []*types.Block {
	return fs.blocks
}

func (fs *ChainDB) Txs() uint64 {
	return fs.txs.Load()
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

func (fs *ChainDB) InitBlockNumber() error {
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

func (fs *ChainDB) CheckPoint() uint64 {
	return fs.checkPoint.Load()
}

func (fs *ChainDB) LastListenBlockNumber() uint64 {
	return fs.lastListenBlockNumber.Load()
}

func (fs *ChainDB) Anchor(n uint64) {
	fs.lastListenBlockNumber.Store(n)
}
