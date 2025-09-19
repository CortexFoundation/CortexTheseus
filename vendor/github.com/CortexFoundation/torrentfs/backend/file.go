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
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/event"
	"github.com/CortexFoundation/CortexTheseus/log"
)

func (tm *TorrentManager) ExistsOrActive(ctx context.Context, ih string, rawSize uint64) (bool, uint64, mclock.AbsTime, error) {
	availableMeter.Mark(1)

	// Validate infohash format early to prevent unnecessary processing.
	if !common.IsHexAddress(ih) {
		return false, 0, 0, errors.New("invalid infohash format")
	}

	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	// Check if the torrent is active in the manager.
	t := tm.getTorrent(ih)
	if t == nil {
		// If not active, check if the torrent directory exists on disk.
		dir := filepath.Join(tm.dataDir, ih)
		if _, err := os.Stat(dir); err == nil {
			// Torrent files exist but are not managed.
			return true, 0, 0, ErrInactiveTorrent
		}
		// Torrent is neither active nor on disk.
		return false, 0, 0, ErrInactiveTorrent
	}

	// The torrent is active. Now check its status.
	if !t.Ready() {
		// If the torrent is not ready, check if it has info.
		if t.Torrent.Info() == nil {
			return false, 0, 0, ErrTorrentNotFound
		}
		// The torrent is active but still in progress.
		bytesCompleted := uint64(t.Torrent.BytesCompleted())
		age := mclock.Now() - t.Birth()
		return false, bytesCompleted, age, ErrUnfinished
	}

	// The torrent is ready and active.
	bytesCompleted := uint64(t.Torrent.BytesCompleted())
	ok := bytesCompleted <= rawSize
	age := mclock.Now() - t.Birth()

	return ok, bytesCompleted, age, nil
}

func (tm *TorrentManager) GetFile(ctx context.Context, infohash, subpath string) ([]byte, *event.TypeMux, error) {
	getfileMeter.Mark(1)
	if tm.metrics {
		// Use a defer statement to ensure the timer is always stopped.
		defer func(start time.Time) { tm.updates += time.Since(start) }(time.Now())
	}

	// Early validation and normalization to reduce nested logic.
	if !common.IsHexAddress(infohash) {
		return nil, nil, errors.New("invalid infohash format")
	}

	ih := strings.TrimPrefix(strings.ToLower(infohash), common.Prefix)
	sp := strings.Trim(subpath, "/")

	log.Debug("Get File", "dir", tm.dataDir, "infohash", ih, "subpath", sp)

	var (
		mu   *event.TypeMux
		err  error
		data []byte
	)

	// Check if the torrent is active and ready.
	if t := tm.getTorrent(ih); t != nil {
		if !t.Ready() {
			return nil, t.Mux(), ErrUnfinished
		}

		// Lock the torrent for reading to ensure data integrity during file access.
		t.RLock()
		defer t.RUnlock()
		mu = t.Mux()
	}

	// Construct the full file path.
	filePath := filepath.Join(tm.dataDir, ih, sp)

	diskReadMeter.Mark(1)

	// Use file cache if active, otherwise fall back to local read.
	if tm.fc != nil && tm.fc.Active() {
		start := mclock.Now()
		data, err = tm.fc.ReadFile(filePath)
		if err == nil {
			elapsed := time.Duration(mclock.Now() - start)
			log.Debug("Loaded data from file cache", "infohash", ih, "file", filePath, "elapsed", common.PrettyDuration(elapsed))
		}
	} else {
		data, err = os.ReadFile(filePath)
	}

	// The named return values 'data' and 'mu' are handled automatically.
	return data, mu, err
}
