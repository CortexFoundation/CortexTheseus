// Copyright 2020 The CortexTheseus Authors
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

package caffe

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/anacrolix/torrent"
)

type Torrent struct {
	*torrent.Torrent
	//maxEstablishedConns int
	//minEstablishedConns int
	//currentConns        int
	bytesRequested atomic.Int64
	//bytesLimitation int64
	//bytesCompleted int64
	//bytesMissing        int64
	status   atomic.Int32
	infohash string
	filepath string
	cited    atomic.Int32
	//weight     int
	//loop       int
	maxPieces atomic.Int32
	//isBoosting bool
	//fast  bool
	start mclock.AbsTime
	//ch    chan bool
	wg sync.WaitGroup

	lock sync.RWMutex

	closeAll chan any

	taskCh chan task

	slot int

	startOnce sync.Once
	stopOnce  sync.Once

	spec *torrent.TorrentSpec

	//jobCh chan bool
}

type task struct {
	start int
	end   int
}

func NewTorrent(t *torrent.Torrent, requested int64, ih string, path string, slot int, spec *torrent.TorrentSpec) *Torrent {
	tor := Torrent{
		Torrent: t,
		//bytesRequested: requested,
		//status:   torrentPending,
		infohash: ih,
		filepath: path,
		start:    mclock.Now(),
		taskCh:   make(chan task, 8),
		closeAll: make(chan any),
		slot:     slot,
		spec:     spec,
	}

	tor.bytesRequested.Store(requested)
	tor.status.Store(TorrentPending)

	return &tor
}

// Find out the start and end
func (t *Torrent) download(p int) error {
	var s, e int
	s = (t.Torrent.NumPieces() * t.slot) / params.Bucket
	s = s - p>>1
	if s < 0 {
		s = 0
	}

	if t.Torrent.NumPieces() < s+p {
		s = t.Torrent.NumPieces() - p
	}

	e = s + p

	if t.taskCh != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		select {
		case t.taskCh <- task{s, e}:
			//log.Debug(ScaleBar(s, e, t.Torrent.NumPieces()), "ih", t.InfoHash(), "slot", t.slot, "s", s, "e", e, "p", p, "total", t.Torrent.NumPieces())
		case <-ctx.Done():
			return ctx.Err()
		case <-t.closeAll:
		}
	} else {
		return errors.New("task channel is nil")
	}
	return nil
}

func (t *Torrent) run() bool {
	t.Lock()
	defer t.Unlock()

	if t.Info() != nil {
		t.status.Store(TorrentRunning)
	} else {
		log.Warn("Task listener not ready", "ih", t.InfoHash())
		return false
	}

	return true
}

func (t *Torrent) listen() {
	defer t.wg.Done()

	log.Debug("Task listener started", "ih", t.InfoHash())

	for {
		select {
		case task := <-t.taskCh:
			t.Torrent.DownloadPieces(task.start, task.end)
		case <-t.closeAll:
			log.Debug("Task listener stopped", "ih", t.InfoHash())
			return
		}
	}
}
