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
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/types"
	bolt "go.etcd.io/bbolt"
	"time"
)

func (fs *ChainDB) Files() []*types.FileInfo {
	return fs.files
}

func (fs *ChainDB) NewFileInfo(fileMeta *types.FileMeta) *types.FileInfo {
	ret := &types.FileInfo{Meta: fileMeta, LeftSize: fileMeta.RawSize}
	return ret
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
