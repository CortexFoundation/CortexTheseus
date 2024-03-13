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
	"errors"
	"strconv"
	"strings"

	//lru "github.com/hashicorp/golang-lru"

	"github.com/CortexFoundation/CortexTheseus/log"
	bolt "go.etcd.io/bbolt"
)

func (fs *ChainDB) Torrents() map[string]uint64 {
	return fs.torrents
}

// SetTorrent is for recording torrent latest status
func (fs *ChainDB) SetTorrentProgress(ih string, size uint64) (bool, uint64, error) {
	fs.lock.Lock()
	defer fs.lock.Unlock()

	if s, ok := fs.torrents[ih]; ok {
		if s >= size {
			return false, s, nil
		}
	}
	if err := fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(TORRENT_ + fs.version))
		if err != nil {
			return err
		}
		v := buk.Get([]byte(ih))

		if v == nil {
			err = buk.Put([]byte(ih), uint64ToHex(size))
		} else {
			s, err := strconv.ParseUint(string(v), 16, 64)
			if err != nil {
				return err
			}
			if size > s {
				err = buk.Put([]byte(ih), uint64ToHex(size))
			} else {
				size = s
			}
		}

		return err
	}); err != nil {
		return false, 0, err
	}

	fs.torrents[ih] = size

	log.Debug("File status has been changed", "ih", ih, "size", size, "count", len(fs.torrents))

	return true, size, nil
}

// GetTorrent return the torrent status by uint64, if return 0 for torrent not exist
func (fs *ChainDB) GetTorrentProgress(ih string) (progress uint64, err error) {
	fs.lock.RLock()
	defer fs.lock.RUnlock()

	//TODO
	ih = strings.ToLower(ih)

	if s, ok := fs.torrents[ih]; ok {
		return s, nil
	}
	cb := func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte(TORRENT_ + fs.version))
		if buk == nil {
			return errors.New("root bucket not exist")
		}

		v := buk.Get([]byte(ih))

		if v == nil {
			return errors.New("No torrent record found")
		}

		s, err := strconv.ParseUint(string(v), 16, 64)
		if err != nil {
			return err
		}

		progress = s

		return nil
	}
	if err := fs.db.View(cb); err != nil {
		return 0, err
	}

	return progress, nil
}

func (fs *ChainDB) InitTorrents() (map[string]uint64, error) {
	err := fs.db.Update(func(tx *bolt.Tx) error {
		if buk, err := tx.CreateBucketIfNotExists([]byte(TORRENT_ + fs.version)); err != nil {
			return err
		} else {
			c := buk.Cursor()
			for k, v := c.First(); k != nil; k, v = c.Next() {
				size, err := strconv.ParseUint(string(v), 16, 64)
				if err != nil {
					return err
				}
				fs.torrents[string(k)] = size
			}
			log.Debug("Torrent initializing ... ...", "torrents", len(fs.torrents))
			return nil
		}
	})
	if err != nil {
		return nil, err
	}
	return fs.torrents, nil
}
