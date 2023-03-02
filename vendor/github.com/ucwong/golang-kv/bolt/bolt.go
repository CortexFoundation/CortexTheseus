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

package bolt

import (
	"bytes"
	//"fmt"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/ucwong/go-ttlmap"
	"github.com/ucwong/golang-kv/common"

	bolt "go.etcd.io/bbolt"
)

type Bolt struct {
	engine  *bolt.DB
	ttl_map *ttlmap.Map

	once sync.Once
}

const GLOBAL = "m41gA7omIWU4s"

type BoltOption func(bolt.Options) bolt.Options

func Open(path string, opts ...BoltOption) *Bolt {
	//if len(path) == 0 {
	path = filepath.Join(path, common.GLOBAL_SPACE, ".bolt")
	err := os.MkdirAll(path, 0777) //os.FileMode(os.ModePerm))
	if err != nil {
		fmt.Println(err)
		return nil
	}
	//}
	b := &Bolt{}

	var option bolt.Options
	for _, opt := range opts {
		option = opt(option)
	}
	if db, err := bolt.Open(filepath.Join(path, ".bolt"), 0777, &option); err == nil {
		b.engine = db
	} else {
		//panic(err)
		return nil
	}

	options := &ttlmap.Options{
		InitialCapacity: 1024 * 1024,
		OnWillExpire: func(key string, item ttlmap.Item) {
			//fmt.Printf("expired: [%s=%v]\n", key, item.Value())
			//b.Del([]byte(key))
		},
		OnWillEvict: func(key string, item ttlmap.Item) {
			//fmt.Printf("evicted: [%s=%v]\n", key, item.Value())
			b.engine.Update(func(tx *bolt.Tx) error {
				if buk := tx.Bucket([]byte(GLOBAL)); buk != nil {
					return buk.Delete([]byte(key))
				}
				return nil
			})
		},
	}
	b.ttl_map = ttlmap.New(options)
	return b
}

func (b *Bolt) Get(k []byte) (v []byte) {
	if item, err := b.ttl_map.Get(string(k)); err == nil {
		return []byte(item.Value().(string))
	}

	b.engine.View(func(tx *bolt.Tx) error {
		if buk := tx.Bucket([]byte(GLOBAL)); buk != nil {
			v = buk.Get(k)
		}
		return nil
	})
	return
}

func (b *Bolt) Set(k, v []byte) (err error) {
	//if _, err = b.ttl_map.Delete(string(k)); err != nil {
	//	return
	//}

	err = b.engine.Update(func(tx *bolt.Tx) error {
		buk, e := tx.CreateBucketIfNotExists([]byte(GLOBAL))
		if e != nil {
			return e
		}
		return buk.Put(k, v)
	})
	return
}

func (b *Bolt) Del(k []byte) (err error) {
	if _, err = b.ttl_map.Delete(string(k)); err != nil {
		return
	}

	err = b.engine.Update(func(tx *bolt.Tx) error {
		if buk := tx.Bucket([]byte(GLOBAL)); buk != nil {
			return buk.Delete(k)
		}
		return nil
	})

	return
}

func (b *Bolt) Prefix(prefix []byte) (res [][]byte) {
	b.engine.View(func(tx *bolt.Tx) error {
		if buk := tx.Bucket([]byte(GLOBAL)); buk != nil {
			c := buk.Cursor()
			for k, v := c.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = c.Next() {
				res = append(res, common.SafeCopy(nil, v))
			}
		}

		return nil
	})

	return
}

func (b *Bolt) Suffix(suffix []byte) (res [][]byte) {
	b.engine.View(func(tx *bolt.Tx) error {
		if buk := tx.Bucket([]byte(GLOBAL)); buk != nil {
			buk.ForEach(func(k, v []byte) error {
				if bytes.HasSuffix(k, suffix) {
					res = append(res, common.SafeCopy(nil, v))
				}
				return nil
			})
		}

		return nil
	})

	return
}

func (b *Bolt) Scan() (res [][]byte) {
	b.engine.View(func(tx *bolt.Tx) error {
		if buk := tx.Bucket([]byte(GLOBAL)); buk != nil {
			buk.ForEach(func(k, v []byte) error {
				res = append(res, common.SafeCopy(nil, v))
				return nil
			})
		}
		return nil
	})

	return
}

func (b *Bolt) SetTTL(k, v []byte, expire time.Duration) (err error) {
	err = b.ttl_map.Set(string(k), ttlmap.NewItem(string(v), ttlmap.WithTTL(expire)), nil)
	if err != nil {
		return
	}
	err = b.engine.Update(func(tx *bolt.Tx) error {
		buk, e := tx.CreateBucketIfNotExists([]byte(GLOBAL))
		if e != nil {
			return e
		}
		return buk.Put(k, v)
	})

	if err != nil {
		// TODO
		b.ttl_map.Delete(string(k))
	}

	return
}

func (b *Bolt) Range(start, limit []byte) (res [][]byte) {
	b.engine.View(func(tx *bolt.Tx) error {
		if buk := tx.Bucket([]byte(GLOBAL)); buk != nil {
			c := buk.Cursor()
			for k, v := c.Seek(start); k != nil && bytes.Compare(start, k) <= 0; k, v = c.Next() {
				if bytes.Compare(limit, k) > 0 {
					res = append(res, common.SafeCopy(nil, v))
				} else {
					break
				}
			}
		}

		return nil
	})
	return
}

func (b *Bolt) Close() error {
	b.once.Do(func() {
		b.ttl_map.Drain()
	})
	return b.engine.Close()
}

func (b *Bolt) BatchSet(kvs map[string][]byte) error {
	return b.engine.Batch(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(GLOBAL))
		for k, v := range kvs {
			if err := bucket.Put([]byte(k), v); err != nil {
				return err
			}
		}
		return nil
	})
}

func (b *Bolt) Name() string {
	return "bolt"
}
