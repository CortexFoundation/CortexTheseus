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

package pebble

import (
	"bytes"
	//"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/cockroachdb/pebble"
	"github.com/ucwong/go-ttlmap"
	"github.com/ucwong/golang-kv/common"
)

type Pebble struct {
	engine  *pebble.DB
	ttl_map *ttlmap.Map
	wb      *pebble.Batch
	once    sync.Once
}

type PebbleOption func(pebble.Options) pebble.Options

func Open(path string, opts ...PebbleOption) *Pebble {
	path = filepath.Join(path, common.GLOBAL_SPACE, ".pebble")
	db := &Pebble{}
	option := pebble.Options{}
	for _, op := range opts {
		option = op(option)
	}
	peb, err := pebble.Open(path, &option)
	if err != nil {
		return nil
	}
	db.engine = peb
	db.wb = new(pebble.Batch)

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

func (peb *Pebble) Get(k []byte) (v []byte) {
	item, err := peb.ttl_map.Get(string(k))
	if err == nil {
		return []byte(item.Value().(string))
	}

	v, closer, err := peb.engine.Get(k)
	if err == nil {
		defer closer.Close()
	}
	return
}

func (peb *Pebble) Set(k, v []byte) (err error) {
	//if _, err = peb.ttl_map.Delete(string(k)); err != nil {
	//	return
	//}

	err = peb.engine.Set(k, v, pebble.Sync)
	return
}

func (peb *Pebble) Del(k []byte) (err error) {
	if _, err = peb.ttl_map.Delete(string(k)); err != nil {
		return
	}

	err = peb.engine.Delete(k, nil)
	return
}

func (peb *Pebble) Prefix(k []byte) (res [][]byte) {
	keyUpperBound := func(b []byte) []byte {
		end := make([]byte, len(b))
		copy(end, b)
		for i := len(end) - 1; i >= 0; i-- {
			end[i] = end[i] + 1
			if end[i] != 0 {
				return end[:i+1]
			}
		}
		return nil // no upper-bound
	}
	prefixIterOptions := func(prefix []byte) *pebble.IterOptions {
		return &pebble.IterOptions{
			LowerBound: prefix,
			UpperBound: keyUpperBound(prefix),
		}
	}

	iter := peb.engine.NewIter(prefixIterOptions(k))
	defer iter.Close()
	for iter.First(); iter.Valid(); iter.Next() {
		res = append(res, common.SafeCopy(nil, iter.Value()))
	}
	return
}

func (peb *Pebble) Suffix(k []byte) (res [][]byte) {
	iter := peb.engine.NewIter(nil)
	defer iter.Close()
	for iter.First(); iter.Valid(); iter.Next() {
		if bytes.HasSuffix(iter.Key(), k) {
			res = append(res, common.SafeCopy(nil, iter.Value()))
		}
	}
	return
}

func (peb *Pebble) Range(start, limit []byte) (res [][]byte) {
	iter := peb.engine.NewIter(nil)
	defer iter.Close()
	for iter.SeekGEWithLimit(start, limit); iter.Valid(); iter.Next() {
		if bytes.Compare(limit, iter.Key()) > 0 {
			res = append(res, common.SafeCopy(nil, iter.Value()))
		} else {
			break
		}
	}
	return
}

func (peb *Pebble) Scan() (res [][]byte) {
	iter := peb.engine.NewIter(nil)
	defer iter.Close()
	for iter.First(); iter.Valid(); iter.Next() {
		res = append(res, common.SafeCopy(nil, iter.Value()))
	}
	return
}

func (peb *Pebble) SetTTL(k, v []byte, expire time.Duration) (err error) {
	if err = peb.ttl_map.Set(string(k), ttlmap.NewItem(string(v), ttlmap.WithTTL(expire)), nil); err != nil {
		return
	}

	err = peb.engine.Set(k, v, pebble.Sync)

	if err != nil {
		// TODO
		peb.ttl_map.Delete(string(k))
	}

	return
}

func (peb *Pebble) Close() error {
	peb.once.Do(func() {
		peb.ttl_map.Drain()
	})
	return peb.engine.Close()
}

func (peb *Pebble) BatchSet(kvs map[string][]byte) error {
	for k, v := range kvs {
		peb.wb.Set([]byte(k), v, nil)
	}
	peb.engine.Apply(peb.wb, nil)
	return peb.wb.SyncWait()
}

func (peb *Pebble) Name() string {
	return "pebble"
}
