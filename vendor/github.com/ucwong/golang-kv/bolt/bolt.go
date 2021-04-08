package bolt

import (
	"bytes"
	//"fmt"
	"time"

	"github.com/imkira/go-ttlmap"
	"github.com/ucwong/golang-kv/common"

	bolt "go.etcd.io/bbolt"
)

type Bolt struct {
	engine  *bolt.DB
	ttl_map *ttlmap.Map
}

const GLOBAL = "m41gA7omIWU4s"

func Open(path string) *Bolt {
	//if len(path) == 0 {
	path = path + ".bolt"
	//}
	b := &Bolt{}
	if db, err := bolt.Open(path, 0600, nil); err == nil {
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
	go b.ttl_map.Delete(string(k))

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
	go b.ttl_map.Delete(string(k))

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
			/*c := buk.Cursor()
			for k, v := c.First(); k != nil; k, v = c.Next() {
				if bytes.HasSuffix(k, suffix) {
					res = append(res, common.SafeCopy(nil, v))
				}
			}*/

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
	b.ttl_map.Drain()
	return b.engine.Close()
}
