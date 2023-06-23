// Copyright (C) 2023 ucwong
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

package rosedb

import (
	"bytes"
	"path/filepath"
	"sync"
	"time"

	"github.com/rosedblabs/rosedb/v2"
	"github.com/ucwong/go-ttlmap"
	"github.com/ucwong/golang-kv/common"
)

type RoseDB struct {
	engine  *rosedb.DB
	ttl_map *ttlmap.Map
	once    sync.Once
}

type RoseDBOption func(rosedb.Options) rosedb.Options

func Open(path string, opts ...RoseDBOption) *RoseDB {
	path = filepath.Join(path, common.GLOBAL_SPACE, ".rose")
	db := &RoseDB{}
	option := rosedb.DefaultOptions //opt.Options{OpenFilesCacheCapacity: 32}
	for _, op := range opts {
		option = op(option)
	}
	option.DirPath = path
	rdb, err := rosedb.Open(option)
	if err != nil {
		return nil
	}
	db.engine = rdb

	options := &ttlmap.Options{
		InitialCapacity: 1024 * 1024,
		OnWillExpire: func(key string, item ttlmap.Item) {
			//fmt.Printf("expired: [%s=%v]\n", key, item.Value())
			//b.Del([]byte(key))
		},
		OnWillEvict: func(key string, item ttlmap.Item) {
			//fmt.Printf("evicted: [%s=%v]\n", key, item.Value())
			//db.Del([]byte(key))
			db.engine.Delete([]byte(key))
		},
	}
	db.ttl_map = ttlmap.New(options)
	return db
}

func (rdb *RoseDB) Get(k []byte) (v []byte) {
	item, err := rdb.ttl_map.Get(string(k))
	if err == nil {
		return []byte(item.Value().(string))
	}

	v, _ = rdb.engine.Get(k)
	return
}

func (rdb *RoseDB) Set(k, v []byte) (err error) {
	//if _, err = rdb.ttl_map.Delete(string(k)); err != nil {
	//	return
	//}

	err = rdb.engine.Put(k, v)
	return
}

func (rdb *RoseDB) Del(k []byte) (err error) {
	if _, err = rdb.ttl_map.Delete(string(k)); err != nil {
		return
	}

	err = rdb.engine.Delete(k)
	return
}

func (rdb *RoseDB) Prefix(k []byte) (res [][]byte) {
	iterOptions := rosedb.DefaultIteratorOptions
	iterOptions.Prefix = k
	iter := rdb.engine.NewIterator(iterOptions)
	defer iter.Close()
	for ; iter.Valid(); iter.Next() {
		val, _ := iter.Value()
		res = append(res, common.SafeCopy(nil, val))
	}
	return
}

func (rdb *RoseDB) Suffix(k []byte) (res [][]byte) {
	iterOptions := rosedb.DefaultIteratorOptions
	iter := rdb.engine.NewIterator(iterOptions)
	defer iter.Close()
	for ; iter.Valid(); iter.Next() {
		if bytes.HasSuffix(iter.Key(), k) {
			val, _ := iter.Value()
			res = append(res, common.SafeCopy(nil, val))
		}
	}
	return
}

func (rdb *RoseDB) Range(start, limit []byte) (res [][]byte) {
	iterOptions := rosedb.DefaultIteratorOptions
	iter := rdb.engine.NewIterator(iterOptions)
	defer iter.Close()
	for iter.Seek(start); iter.Valid(); iter.Next() {
		if bytes.Compare(limit, iter.Key()) > 0 && bytes.Compare(start, iter.Key()) <= 0 {
			val, _ := iter.Value()
			res = append(res, common.SafeCopy(nil, val))
		} else {
			break
		}
	}
	return
}

func (rdb *RoseDB) Scan() (res [][]byte) {
	iterOptions := rosedb.DefaultIteratorOptions
	iter := rdb.engine.NewIterator(iterOptions)
	defer iter.Close()
	for ; iter.Valid(); iter.Next() {
		val, _ := iter.Value()
		res = append(res, common.SafeCopy(nil, val))
	}
	return
}

func (rdb *RoseDB) SetTTL(k, v []byte, expire time.Duration) (err error) {
	if err = rdb.ttl_map.Set(string(k), ttlmap.NewItem(string(v), ttlmap.WithTTL(expire)), nil); err != nil {
		return
	}

	err = rdb.engine.Put(k, v)

	if err != nil {
		// TODO
		rdb.ttl_map.Delete(string(k))
	}

	return
}

func (rdb *RoseDB) Close() error {
	rdb.once.Do(func() {
		rdb.ttl_map.Drain()
	})
	return rdb.engine.Close()
}

func (rdb *RoseDB) BatchSet(kvs map[string][]byte) error {
	wb := rdb.engine.NewBatch(rosedb.DefaultBatchOptions)
	for k, v := range kvs {
		wb.Put([]byte(k), v)
	}
	return wb.Commit()
}

func (rdb *RoseDB) Name() string {
	return "rosedb"
}
