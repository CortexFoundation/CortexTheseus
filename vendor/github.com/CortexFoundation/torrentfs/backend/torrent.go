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

package backend

import (
	"errors"
	//"bytes"
	//"context"
	"context"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/anacrolix/torrent"
	//"github.com/anacrolix/torrent/metainfo"
	//"github.com/anacrolix/torrent/storage"
	"github.com/CortexFoundation/torrentfs/params"
)

const (
	torrentPending = iota + 1
	torrentPaused
	torrentRunning
	torrentSeeding
	torrentStopping
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
	tor.status.Store(torrentPending)

	return &tor
}

func (t *Torrent) QuotaFull() bool {
	//t.RLock()
	//defer t.RUnlock()

	return t.Info() != nil && t.bytesRequested.Load() >= t.Length()
}

func (t *Torrent) Spec() *torrent.TorrentSpec {
	return t.spec
}

func (t *Torrent) Birth() mclock.AbsTime {
	return t.start
}

func (t *Torrent) Lock() {
	t.lock.Lock()
}

func (t *Torrent) Unlock() {
	t.lock.Unlock()
}

func (t *Torrent) RLock() {
	t.lock.RLock()
}

func (t *Torrent) RUnlock() {
	t.lock.RUnlock()
}

/*func (t *Torrent) BytesLeft() int64 {
	if t.bytesRequested < t.bytesCompleted {
		return 0
	}
	return t.bytesRequested - t.bytesCompleted
}*/

func (t *Torrent) InfoHash() string {
	return t.infohash
}

func (t *Torrent) Status() int {
	return int(t.status.Load())
}

func (t *Torrent) Cited() int32 {
	return t.cited.Load()
}

func (t *Torrent) CitedInc() {
	t.cited.Add(1)
}

func (t *Torrent) CitedDec() {
	t.cited.Add(-1)
}

func (t *Torrent) BytesRequested() int64 {
	return t.bytesRequested.Load()
}

func (t *Torrent) SetBytesRequested(bytesRequested int64) {
	//t.Lock()
	//defer t.Unlock()
	//t.bytesRequested = bytesRequested
	t.bytesRequested.Store(bytesRequested)
}

func (t *Torrent) Ready() bool {
	if _, ok := params.BadFiles[t.InfoHash()]; ok {
		return false
	}

	ret := t.IsSeeding()
	if !ret {
		//log.Debug("Not ready", "ih", t.InfoHash(), "status", t.status, "seed", t.Torrent.Seeding(), "seeding", torrentSeeding)
	}

	return ret
}

func (t *Torrent) WriteTorrent() error {
	t.Lock()
	defer t.Unlock()
	if _, err := os.Stat(filepath.Join(t.filepath, TORRENT)); err == nil {
		//t.Pause()
		return nil
	}

	if f, err := os.OpenFile(filepath.Join(t.filepath, TORRENT), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0777); err == nil {
		defer f.Close()
		log.Debug("Write seed file", "path", t.filepath)
		if err := t.Metainfo().Write(f); err != nil {
			log.Warn("Write seed error", "err", err)
			return err
		}
	} else {
		log.Warn("Create Path error", "err", err)
		return err
	}

	return nil
}

//func (t *Torrent) BoostOff() {
//t.isBoosting = false
//}

func (t *Torrent) Seed() bool {
	//t.lock.Lock()
	//defer t.lock.Unlock()

	if t.Torrent.Info() == nil {
		log.Debug("Nas info is nil", "ih", t.InfoHash())
		return false
	}
	if t.status.Load() == torrentSeeding {
		//log.Debug("Nas status is", "status", t.status, "ih", t.InfoHash())
		return true
	}
	//if t.currentConns <= t.minEstablishedConns {
	//t.setCurrentConns(t.maxEstablishedConns)
	//t.Torrent.SetMaxEstablishedConns(t.currentConns)
	//}
	if t.Torrent.Seeding() {
		//t.Lock()
		//defer t.Unlock()

		//t.status = torrentSeeding
		t.status.Store(torrentSeeding)
		t.stopListen()

		elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
		//if active, ok := params.GoodFiles[t.InfoHash()]; !ok {
		//	log.Info("New active nas found", "ih", t.InfoHash(), "ok", ok, "active", active, "size", common.StorageSize(t.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "status", t.status, "elapsed", common.PrettyDuration(elapsed))
		//} else {
		log.Info("Imported new nas segment", "ih", t.InfoHash(), "size", common.StorageSize(t.Torrent.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "status", t.status.Load(), "elapsed", common.PrettyDuration(elapsed), "speed", common.StorageSize(float64(t.Torrent.BytesCompleted()*1000*1000*1000)/float64(elapsed)).String()+"/s")
		//}

		return true
	}

	return false
}

func (t *Torrent) IsSeeding() bool {
	//t.RLock()
	//defer t.RUnlock()

	return t.status.Load() == torrentSeeding // && t.Torrent.Seeding()
}

func (t *Torrent) Pause() {
	//t.Lock()
	//defer t.Unlock()
	//if t.currentConns > t.minEstablishedConns {
	//t.setCurrentConns(t.minEstablishedConns)
	//t.Torrent.SetMaxEstablishedConns(t.minEstablishedConns)
	//}
	if t.status.Load() != torrentPaused {
		//t.status = torrentPaused
		t.status.Store(torrentPaused)
		//t.maxPieces = 0 //t.minEstablishedConns
		t.maxPieces.Store(0)
		t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
	}
}

func (t *Torrent) Paused() bool {
	//t.RLock()
	//defer t.RUnlock()

	return t.status.Load() == torrentPaused
}

func (t *Torrent) Leech() error {
	// Make sure the torrent info exists
	if t.Torrent.Info() == nil {
		return errors.New("info is nil")
	}

	if t.status.Load() != torrentRunning {
		return errors.New("nas is not running")
	}

	if t.Torrent.BytesMissing() == 0 {
		return nil
	}

	limitPieces := int((t.bytesRequested.Load()*int64(t.Torrent.NumPieces()) + t.Length() - 1) / t.Length())
	if limitPieces > t.Torrent.NumPieces() {
		limitPieces = t.Torrent.NumPieces()
	}

	//t.Lock()
	//defer t.Unlock()

	if limitPieces > int(t.maxPieces.Load()) {
		if err := t.download(limitPieces); err == nil {
			//t.maxPieces = limitPieces
			t.maxPieces.Store(int32(limitPieces))
		} else {
			return err
		}
	}

	return nil
}

// Find out the start and end
func (t *Torrent) download(p int) error {
	var s, e int
	s = (t.Torrent.NumPieces() * t.slot) / bucket
	s = s - p/2
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
			log.Info(ScaleBar(s, e, t.Torrent.NumPieces()), "ih", t.InfoHash(), "slot", t.slot, "s", s, "e", e, "p", p, "total", t.Torrent.NumPieces())
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
		//t.status = torrentRunning
		t.status.Store(torrentRunning)
	} else {
		log.Warn("Task listener not ready", "ih", t.InfoHash())
		return false
	}

	return true
}

func (t *Torrent) listen() {
	defer t.wg.Done()

	log.Info("Task listener started", "ih", t.InfoHash())

	for {
		select {
		case task := <-t.taskCh:
			t.Torrent.DownloadPieces(task.start, task.end)
		case <-t.closeAll:
			log.Info("Task listener stopped", "ih", t.InfoHash())
			return
		}
	}
}

func (t *Torrent) Running() bool {
	//t.RLock()
	//defer t.RUnlock()

	return t.status.Load() == torrentRunning
}

func (t *Torrent) Pending() bool {
	//t.RLock()
	//defer t.RUnlock()

	return t.status.Load() == torrentPending
}

func (t *Torrent) Stopping() bool {
	//t.RLock()
	//defer t.RUnlock()

	return t.status.Load() == torrentStopping
}

func (t *Torrent) Start() error {
	if !t.run() {
		return errors.New("nas run failed")
	}

	t.startOnce.Do(func() {
		t.wg.Add(1)
		go t.listen()
	})
	return nil
}

func (t *Torrent) Stop() {
	t.Lock()
	defer t.Unlock()

	defer t.Torrent.Drop()

	if t.Status() != torrentStopping {
		log.Info(ProgressBar(t.BytesCompleted(), t.Torrent.Length(), ""), "ih", t.InfoHash(), "total", common.StorageSize(t.Torrent.Length()), "req", common.StorageSize(t.BytesRequested()), "finish", common.StorageSize(t.Torrent.BytesCompleted()), "status", t.Status(), "cited", t.Cited())
		//t.status = torrentStopping
		t.status.Store(torrentStopping)
	}
}

func (t *Torrent) stopListen() {
	t.stopOnce.Do(func() {
		t.Lock()
		defer t.Unlock()

		close(t.closeAll)
		t.wg.Wait()

		t.taskCh = nil

		log.Info("Nas listener stopped", "ih", t.InfoHash(), "status", t.Status())
	})
}

func (t *Torrent) Close() {
	t.Lock()
	defer t.Unlock()

	defer t.Torrent.Drop()

	log.Info("Nas closed", "ih", t.InfoHash(), "total", common.StorageSize(t.Torrent.Length()), "req", common.StorageSize(t.BytesRequested()), "finish", common.StorageSize(t.Torrent.BytesCompleted()), "status", t.Status(), "cited", t.Cited())
	t = nil
}
