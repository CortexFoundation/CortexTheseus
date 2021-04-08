package leveldb

import (
	"bytes"
	//"fmt"
	"time"

	"github.com/imkira/go-ttlmap"
	"github.com/syndtr/goleveldb/leveldb"
	"github.com/syndtr/goleveldb/leveldb/errors"
	"github.com/syndtr/goleveldb/leveldb/opt"
	"github.com/syndtr/goleveldb/leveldb/util"
	"github.com/ucwong/golang-kv/common"
)

type LevelDB struct {
	engine  *leveldb.DB
	ttl_map *ttlmap.Map
}

func Open(path string) *LevelDB {
	//if len(path) == 0 {
	path = path + ".leveldb"
	//}
	db := &LevelDB{}
	ldb, err := leveldb.OpenFile(path, &opt.Options{OpenFilesCacheCapacity: 32})
	if _, iscorrupted := err.(*errors.ErrCorrupted); iscorrupted {
		ldb, err = leveldb.RecoverFile(path, nil)
	}
	if err != nil {
		//panic(err)
		return nil
	}
	db.engine = ldb

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

	v1, _ := ldb.engine.Get(k, nil)
	v = v1
	return
}

func (ldb *LevelDB) Set(k, v []byte) (err error) {
	go ldb.ttl_map.Delete(string(k))

	err = ldb.engine.Put(k, v, nil)
	return
}

func (ldb *LevelDB) Del(k []byte) (err error) {
	go ldb.ttl_map.Delete(string(k))

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
	err = ldb.ttl_map.Set(string(k), ttlmap.NewItem(string(v), ttlmap.WithTTL(expire)), nil)
	if err != nil {
		return
	}

	err = ldb.engine.Put(k, v, nil)

	return
}

func (ldb *LevelDB) Close() error {
	ldb.ttl_map.Drain()
	return ldb.engine.Close()
}
