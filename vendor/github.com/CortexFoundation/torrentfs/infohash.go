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
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>

package torrentfs

import (
	"context"
	"strings"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/ucwong/go-ttlmap"
)

// Available is used to check the file status
func (fs *TorrentFS) wakeup(ctx context.Context, ih string) error {
	if p, e := fs.progress(ih); e == nil {
		return fs.storage().Search(ctx, ih, p)
	} else {
		return e
	}
}

func (fs *TorrentFS) encounter(ih string) {
	if !fs.worm.Contains(ih) {
		fs.worm.Add(ih)
	}
}

func (fs *TorrentFS) progress(ih string) (uint64, error) {
	return fs.monitor.DB().GetTorrentProgress(ih)
}

// Download is used to download file with request, broadcast when not found locally
func (fs *TorrentFS) download(ctx context.Context, ih string, request uint64) error {
	ih = strings.ToLower(ih)
	_, p, err := fs.monitor.DB().SetTorrentProgress(ih, request)
	if err != nil {
		return err
	}
	if exist, _, _, _ := fs.storage().ExistsOrActive(ctx, ih, request); !exist {
		fs.wg.Add(1)
		go func(ih string, p uint64) {
			defer fs.wg.Done()
			s := fs.broadcast(ih, p)
			if s {
				log.Debug("Nas "+params.ProtocolVersionStr+" tunnel", "ih", ih, "request", common.StorageSize(float64(p)), "tunnel", fs.tunnel.Len(), "peers", fs.Neighbors())
			}
		}(ih, p)
	}
	// local search
	if err := fs.storage().Search(ctx, ih, p); err != nil {
		return err
	}

	return nil
}

func (fs *TorrentFS) collapse(ih string, rawSize uint64) bool {
	if s, err := fs.tunnel.Get(ih); err == nil && s.Value().(uint64) >= rawSize {
		return true
	}

	return false
}

func (fs *TorrentFS) traverse(ih string, rawSize uint64) error {
	if err := fs.tunnel.Set(ih, ttlmap.NewItem(rawSize, ttlmap.WithTTL(60*time.Second)), nil); err == nil {
		log.Trace("Wormhole traverse", "ih", ih, "size", common.StorageSize(rawSize))
	} else {
		return err
	}
	return nil
}

func (fs *TorrentFS) broadcast(ih string, rawSize uint64) bool {
	if !common.IsHexAddress(ih) {
		return false
	}

	//if s, err := fs.tunnel.Get(ih); err == nil && s.Value().(uint64) >= rawSize {
	if fs.collapse(ih, rawSize) {
		return false
	}

	//fs.tunnel.Set(ih, ttlmap.NewItem(rawSize, ttlmap.WithTTL(60*time.Second)), nil)
	if err := fs.traverse(ih, rawSize); err != nil {
		return false
	}

	return true
}

func (fs *TorrentFS) record(id string) {
	if !fs.history.Contains(id) {
		fs.history.Add(id)
	}
}
