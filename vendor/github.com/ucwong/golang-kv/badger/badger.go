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

package badger

import (
	"bytes"
	//"fmt"
	"path/filepath"
	"time"

	badger "github.com/dgraph-io/badger/v3"
	//"github.com/dgraph-io/badger/v3/options"
	"github.com/ucwong/golang-kv/common"
)

type Badger struct {
	engine *badger.DB
	wb     *badger.WriteBatch
}

type BadgerOption func(badger.Options) badger.Options

func Open(path string, opts ...BadgerOption) *Badger {
	//if len(path) == 0 {
	path = filepath.Join(path, common.GLOBAL_SPACE, ".badger")
	//}
	b := &Badger{}
	options := badger.DefaultOptions(path)
	for _, opt := range opts {
		options = opt(options)
	}
	if bg, err := badger.Open(options); err == nil {
		b.engine = bg
		b.wb = bg.NewWriteBatch()
	} else {
		//panic(err)
		return nil
	}
	return b
}

func (b *Badger) Get(k []byte) (v []byte) {
	b.engine.View(func(txn *badger.Txn) error {
		if item, err := txn.Get(k); err == nil {
			if val, err := item.ValueCopy(nil); err == nil {
				v = val
			}
		}
		return nil
	})
	return
}

func (b *Badger) Set(k, v []byte) (err error) {
	err = b.engine.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte(k), []byte(v))
	})
	return
}

func (b *Badger) Del(k []byte) (err error) {
	err = b.engine.Update(func(txn *badger.Txn) error {
		return txn.Delete(k)
	})
	return
}

func (b *Badger) Prefix(k []byte) (res [][]byte) {
	b.engine.View(func(txn *badger.Txn) error {
		it := txn.NewIterator(badger.DefaultIteratorOptions)
		defer it.Close()
		for it.Seek(k); it.ValidForPrefix(k); it.Next() {
			item := it.Item()
			if val, err := item.ValueCopy(nil); err == nil {
				res = append(res, val)
			}
		}
		return nil
	})
	return
}

func (b *Badger) Suffix(k []byte) (res [][]byte) {
	b.engine.View(func(txn *badger.Txn) error {
		it := txn.NewIterator(badger.DefaultIteratorOptions)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			if bytes.HasSuffix(item.Key(), k) {
				if val, err := item.ValueCopy(nil); err == nil {
					res = append(res, val)
				}
			}
		}
		return nil
	})
	return
}

func (b *Badger) Scan() (res [][]byte) {
	b.engine.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchSize = 128
		it := txn.NewIterator(opts)
		defer it.Close()
		for it.Rewind(); it.Valid(); it.Next() {
			item := it.Item()
			if val, err := item.ValueCopy(nil); err == nil {
				res = append(res, val)
			}
		}
		return nil
	})
	return
}

func (b *Badger) SetTTL(k, v []byte, expire time.Duration) (err error) {
	err = b.engine.Update(func(txn *badger.Txn) error {
		e := badger.NewEntry(k, v).WithTTL(expire)
		return txn.SetEntry(e)
	})
	return
}

func (b *Badger) Range(start, limit []byte) (res [][]byte) {
	b.engine.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchSize = 128
		it := txn.NewIterator(opts)
		defer it.Close()
		for it.Seek(start); it.Valid() && bytes.Compare(start, it.Item().Key()) <= 0; it.Next() {
			if bytes.Compare(limit, it.Item().Key()) > 0 {
				if val, err := it.Item().ValueCopy(nil); err == nil {
					res = append(res, val)
				}
			} else {
				break
			}
		}
		return nil
	})
	return
}

func (b *Badger) Close() error {
	return b.engine.Close()
}

func (b *Badger) BatchSet(kvs map[string][]byte) error {
	for k, v := range kvs {
		b.wb.Set([]byte(k), v)
	}
	return b.wb.Flush()
}
