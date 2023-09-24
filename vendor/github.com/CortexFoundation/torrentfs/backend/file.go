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
	"context"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func (tm *TorrentManager) ExistsOrActive(ctx context.Context, ih string, rawSize uint64) (bool, uint64, mclock.AbsTime, error) {
	availableMeter.Mark(1)

	if !common.IsHexAddress(ih) {
		return false, 0, 0, errors.New("invalid infohash format")
	}

	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	if t := tm.getTorrent(ih); t == nil {
		dir := filepath.Join(tm.DataDir, ih)
		if _, err := os.Stat(dir); err == nil {
			return true, 0, 0, ErrInactiveTorrent
		}
		return false, 0, 0, ErrInactiveTorrent
	} else {
		if !t.Ready() {
			if t.Torrent.Info() == nil {
				return false, 0, 0, ErrTorrentNotFound
			}
			return false, uint64(t.Torrent.BytesCompleted()), mclock.Now() - t.Birth(), ErrUnfinished
		}

		// TODO
		ok := t.Torrent.BytesCompleted() <= int64(rawSize)

		return ok, uint64(t.Torrent.BytesCompleted()), mclock.Now() - t.Birth(), nil
	}
}

func (tm *TorrentManager) GetFile(ctx context.Context, infohash, subpath string) (data []byte, err error) {
	getfileMeter.Mark(1)
	if tm.metrics {
		defer func(start time.Time) { tm.Updates += time.Since(start) }(time.Now())
	}

	if !common.IsHexAddress(infohash) {
		return nil, errors.New("invalid infohash format")
	}

	infohash = strings.TrimPrefix(strings.ToLower(infohash), common.Prefix)
	subpath = strings.TrimPrefix(subpath, "/")
	subpath = strings.TrimSuffix(subpath, "/")

	var key = filepath.Join(infohash, subpath)

	log.Debug("Get File", "dir", tm.DataDir, "key", key)

	if t := tm.getTorrent(infohash); t != nil {
		if !t.Ready() {
			return nil, ErrUnfinished
		}

		// Data protection when torrent is active
		t.RLock()
		defer t.RUnlock()

	}

	diskReadMeter.Mark(1)
	dir := filepath.Join(tm.DataDir, key)
	if tm.fc != nil && tm.fc.Active() {
		start := mclock.Now()
		if data, err = tm.fc.ReadFile(dir); err == nil {
			log.Debug("Load data from file cache", "ih", infohash, "dir", dir, "elapsed", common.PrettyDuration(time.Duration(mclock.Now()-start)))
		}
	} else {
		// local read
		data, err = os.ReadFile(dir)
	}

	return
}
