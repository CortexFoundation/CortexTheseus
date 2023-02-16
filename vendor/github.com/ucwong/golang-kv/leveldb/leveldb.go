// Copyright (C) 2022 ucwong
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>

package leveldb

import (
	"bytes"
	//"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/errors"
	"github.com/syndtr/goleveldb/leveldb/opt"
	"github.com/syndtr/goleveldb/leveldb/util"
	"github.com/ucwong/go-ttlmap"
	"github.com/ucwong/golang-kv/common"
)

type LevelDB struct {
	engine  *leveldb.DB
	ttl_map *ttlmap.Map
	wb      *leveldb.Batch
	once    sync.Once
}

type LevelDBOption func(opt.Options) opt.Options

func Open(path string, opts ...LevelDBOption) *LevelDB {
	//if len(path) == 0 {
	path = filepath.Join(path, common.GLOBAL_SPACE, ".leveldb")
	//}
	db := &LevelDB{}
	option := opt.Options{OpenFilesCacheCapacity: 32}
	for _, op := range opts {
		option = op(option)
	}
	ldb, err := leveldb.OpenFile(path, &option)
	if _, iscorrupted := err.(*errors.ErrCorrupted); iscorrupted {
		ldb, err = leveldb.RecoverFile(path, nil)
	}
	if err != nil {
		//panic(err)
		return nil
	}
	db.engine = ldb
	db.wb = new(leveldb.Batch)

	options := &ttlmap.Options{
		InitialCapacity: 1024 * 1024,
		OnWillExpire: func(key string, item ttlmap.Item) {
			//fmt.Printf("expired: [%s=%v]\n", key, item.Value())
			//b.Del([]byte(key))
		},
		OnWillEvict: func(key string, item ttlmap.Item) {
			//fmt.Printf("evicted: [%s=%v]\n", key, item.Value())
			//db.Del([]byte(key))
			db.engine.Delete([]byte(key), nil)
		},
	}
	db.ttl_map = ttlmap.New(options)
	return db
}

func (ldb *LevelDB) Get(k []byte) (v []byte) {
	item, err := ldb.ttl_map.Get(string(k))
	if err == nil {
		return []byte(item.Value().(string))
	}

	v, _ = ldb.engine.Get(k, nil)
	return
}

func (ldb *LevelDB) Set(k, v []byte) (err error) {
	//if _, err = ldb.ttl_map.Delete(string(k)); err != nil {
	//	return
	//}

	err = ldb.engine.Put(k, v, nil)
	return
}

func (ldb *LevelDB) Del(k []byte) (err error) {
	if _, err = ldb.ttl_map.Delete(string(k)); err != nil {
		return
	}

	err = ldb.engine.Delete(k, nil)
	return
}

func (ldb *LevelDB) Prefix(k []byte) (res [][]byte) {
	iter := ldb.engine.NewIterator(util.BytesPrefix(k), nil)
	defer iter.Release()
	for iter.Next() {
		res = append(res, common.SafeCopy(nil, iter.Value()))
	}
	return
}

func (ldb *LevelDB) Suffix(k []byte) (res [][]byte) {
	iter := ldb.engine.NewIterator(nil, nil)
	defer iter.Release()
	for iter.Next() {
		if bytes.HasSuffix(iter.Key(), k) {
			res = append(res, common.SafeCopy(nil, iter.Value()))
		}
	}
	return
}

func (ldb *LevelDB) Range(start, limit []byte) (res [][]byte) {
	iter := ldb.engine.NewIterator(&util.Range{Start: start, Limit: limit}, nil)
	defer iter.Release()
	for iter.Next() {
		res = append(res, common.SafeCopy(nil, iter.Value()))
	}
	return
}

func (ldb *LevelDB) Scan() (res [][]byte) {
	iter := ldb.engine.NewIterator(nil, nil)
	defer iter.Release()
	for iter.Next() {
		res = append(res, common.SafeCopy(nil, iter.Value()))
	}
	return
}

func (ldb *LevelDB) SetTTL(k, v []byte, expire time.Duration) (err error) {
	if err = ldb.ttl_map.Set(string(k), ttlmap.NewItem(string(v), ttlmap.WithTTL(expire)), nil); err != nil {
		return
	}

	err = ldb.engine.Put(k, v, nil)

	if err != nil {
		// TODO
		ldb.ttl_map.Delete(string(k))
	}

	return
}

func (ldb *LevelDB) Close() error {
	ldb.once.Do(func() {
		ldb.ttl_map.Drain()
	})
	return ldb.engine.Close()
}

func (ldb *LevelDB) BatchSet(kvs map[string][]byte) error {
	for k, v := range kvs {
		ldb.wb.Put([]byte(k), v)
	}
	return ldb.engine.Write(ldb.wb, nil)
}
