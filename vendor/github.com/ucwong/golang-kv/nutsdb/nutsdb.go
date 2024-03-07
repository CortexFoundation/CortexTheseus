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

package nutsdb

import (
	"bytes"
	"os"
	"path/filepath"
	"time"

	nutsdb "github.com/nutsdb/nutsdb"

	"github.com/ucwong/golang-kv/common"
)

type NutsDB struct {
	engine *nutsdb.DB
}

const GLOBAL = "m41gA7omIWU4s"

type NutsDBOption func(nutsdb.Options) nutsdb.Options

func Open(path string, opts ...NutsDBOption) *NutsDB {
	//if len(path) == 0 {
	path = filepath.Join(path, common.GLOBAL_SPACE, ".nuts")
	err := os.MkdirAll(path, 0777) //os.FileMode(os.ModePerm))
	if err != nil {
		return nil
	}
	//}
	b := &NutsDB{}

	var option nutsdb.Options
	for _, opt := range opts {
		option = opt(option)
	}
	//if db, err := nutsdb.Open(nutsdb.DefaultOptions, nutsdb.WithDir(filepath.Join(path, ".nuts"))); err == nil {
	if db, err := nutsdb.Open(nutsdb.DefaultOptions, nutsdb.WithDir(path)); err == nil {
		b.engine = db
	} else {
		//panic(err)
		return nil
	}

	return b
}

func (b *NutsDB) Get(k []byte) (v []byte) {
	b.engine.View(func(tx *nutsdb.Tx) error {
		if e, err := tx.Get(GLOBAL, k); err != nil {
			return err
		} else {
			v = e
		}
		return nil
	})
	return
}

func (b *NutsDB) Set(k, v []byte) (err error) {
	return b.engine.View(func(tx *nutsdb.Tx) error {
		return tx.Put(GLOBAL, k, v, 0)
	})
}

func (b *NutsDB) Del(k []byte) (err error) {
	err = b.engine.Update(func(tx *nutsdb.Tx) error {
		return tx.Delete(GLOBAL, k)
	})

	return
}

func (b *NutsDB) Prefix(prefix []byte) (res [][]byte) {
	b.engine.View(func(tx *nutsdb.Tx) error {
		if entries, err := tx.PrefixScan(GLOBAL, prefix, 25, 100); err != nil {
			return err
		} else {
			for _, entry := range entries {
				res = append(res, common.SafeCopy(nil, entry))
			}
		}

		return nil
	})

	return
}

func (b *NutsDB) Suffix(suffix []byte) (res [][]byte) {
	b.engine.View(func(tx *nutsdb.Tx) error {
		_, entries, err := tx.GetAll(GLOBAL)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			if bytes.HasSuffix(entry, suffix) {
				res = append(res, common.SafeCopy(nil, entry))
			}
		}

		return nil
	})

	return
}

func (b *NutsDB) Scan() (res [][]byte) {
	b.engine.View(func(tx *nutsdb.Tx) error {
		_, entries, err := tx.GetAll(GLOBAL)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			res = append(res, common.SafeCopy(nil, entry))
		}
		return nil
	})

	return
}

func (b *NutsDB) SetTTL(k, v []byte, expire time.Duration) (err error) {
	err = b.engine.Update(func(tx *nutsdb.Tx) error {
		if err := tx.Put(GLOBAL, k, v, uint32(expire.Seconds())); err != nil {
			return err
		}
		return nil
	})

	return
}

func (b *NutsDB) Range(start, limit []byte) (res [][]byte) {
	b.engine.View(func(tx *nutsdb.Tx) error {
		if entries, err := tx.RangeScan(GLOBAL, start, limit); err != nil {
			return err
		} else {
			for _, entry := range entries {
				res = append(res, common.SafeCopy(nil, entry))
			}
		}

		return nil
	})
	return
}

func (b *NutsDB) Close() error {
	return b.engine.Close()
}

func (b *NutsDB) BatchSet(kvs map[string][]byte) error {
	err := b.engine.Update(func(tx *nutsdb.Tx) error {
		for k, v := range kvs {
			if err := tx.Put(GLOBAL, []byte(k), v, 0); err != nil {
				return err
			}
		}
		return nil
	})
	return err
}

func (b *NutsDB) Name() string {
	return "nutsdb"
}
