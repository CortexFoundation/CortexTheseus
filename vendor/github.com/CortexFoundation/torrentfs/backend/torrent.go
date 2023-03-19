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
	//"bytes"
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
	//torrentSleeping
)

type Torrent struct {
	*torrent.Torrent
	//maxEstablishedConns int
	//minEstablishedConns int
	//currentConns        int
	bytesRequested int64
	//bytesLimitation int64
	//bytesCompleted int64
	//bytesMissing        int64
	status   int
	infohash string
	filepath string
	cited    int32
	//weight     int
	//loop       int
	maxPieces int
	//isBoosting bool
	//fast  bool
	start mclock.AbsTime
	//ch    chan bool
	wg sync.WaitGroup

	lock sync.RWMutex

	//closeAll chan any
}

func NewTorrent(t *torrent.Torrent, requested int64, ih string, path string) *Torrent {
	return &Torrent{
		Torrent:        t,
		bytesRequested: requested,
		status:         torrentPending,
		infohash:       ih,
		filepath:       path,
		start:          mclock.Now(),
		//closeAll:       make(chan any),
	}
}

func (t *Torrent) QuotaFull() bool {
	//t.RLock()
	//defer t.RUnlock()

	return t.Info() != nil && t.bytesRequested >= t.Length()
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
	return t.status
}

func (t *Torrent) Cited() int32 {
	return atomic.LoadInt32(&t.cited)
}

func (t *Torrent) CitedInc() {
	atomic.AddInt32(&t.cited, 1)
}

func (t *Torrent) CitedDec() {
	atomic.AddInt32(&t.cited, -1)
}

func (t *Torrent) BytesRequested() int64 {
	//t.RLock()
	//defer t.RUnlock()

	return t.bytesRequested
}

func (t *Torrent) SetBytesRequested(bytesRequested int64) {
	t.Lock()
	defer t.Unlock()
	t.bytesRequested = bytesRequested
}

func (t *Torrent) Ready() bool {
	t.RLock()
	defer t.RUnlock()

	if _, ok := params.BadFiles[t.InfoHash()]; ok {
		return false
	}

	ret := t.IsSeeding()
	if !ret {
		log.Debug("Not ready", "ih", t.InfoHash(), "status", t.status, "seed", t.Torrent.Seeding(), "seeding", torrentSeeding)
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
		log.Debug("Torrent info is nil", "ih", t.InfoHash())
		return false
	}
	if t.status == torrentSeeding {
		log.Debug("Torrent status is", "status", t.status, "ih", t.InfoHash())
		return true
	}
	//if t.currentConns <= t.minEstablishedConns {
	//t.setCurrentConns(t.maxEstablishedConns)
	//t.Torrent.SetMaxEstablishedConns(t.currentConns)
	//}
	if t.Torrent.Seeding() {
		t.Lock()
		t.status = torrentSeeding
		t.Unlock()

		elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
		//if active, ok := params.GoodFiles[t.InfoHash()]; !ok {
		//	log.Info("New active nas found", "ih", t.InfoHash(), "ok", ok, "active", active, "size", common.StorageSize(t.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "peers", t.currentConns, "status", t.status, "elapsed", common.PrettyDuration(elapsed))
		//} else {
		log.Info("Imported new nas segment", "ih", t.InfoHash(), "size", common.StorageSize(t.Torrent.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "status", t.status, "elapsed", common.PrettyDuration(elapsed), "speed", common.StorageSize(float64(t.Torrent.BytesCompleted()*1000*1000*1000)/float64(elapsed)).String()+"/s")
		//}
		return true
	}

	return false
}

func (t *Torrent) IsSeeding() bool {
	return t.status == torrentSeeding && t.Torrent.Seeding()
}

func (t *Torrent) Pause() {
	//if t.currentConns > t.minEstablishedConns {
	//t.setCurrentConns(t.minEstablishedConns)
	//t.Torrent.SetMaxEstablishedConns(t.minEstablishedConns)
	//}
	if t.status != torrentPaused {
		t.status = torrentPaused
		t.maxPieces = 0 //t.minEstablishedConns
		t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
	}
}

func (t *Torrent) Paused() bool {
	return t.status == torrentPaused
}

func (t *Torrent) Start(slot int) {
	// Make sure the torrent info exists
	if t.Torrent.Info() == nil {
		return
	}

	t.Lock()
	defer t.Unlock()

	if t.status != torrentRunning {
		t.status = torrentRunning
	}

	limitPieces := int((t.bytesRequested*int64(t.Torrent.NumPieces()) + t.Length() - 1) / t.Length())
	if limitPieces > t.Torrent.NumPieces() {
		limitPieces = t.Torrent.NumPieces()
	}

	//if limitPieces <= t.maxPieces && t.status == torrentRunning {
	//	return
	//}

	//if t.fast {
	//if t.currentConns <= t.minEstablishedConns {
	//t.setCurrentConns(t.maxEstablishedConns)
	//t.Torrent.SetMaxEstablishedConns(t.currentConns)
	//}
	//} else {
	//	if t.currentConns > t.minEstablishedConns {
	//		t.setCurrentConns(t.minEstablishedConns)
	//		t.Torrent.SetMaxEstablishedConns(t.currentConns)
	//	}
	//}
	if limitPieces > t.maxPieces {
		//t.maxPieces = limitPieces
		if err := t.download(limitPieces, slot); err == nil {
			t.maxPieces = limitPieces
		}
	}
}

// Find out the start and end
func (t *Torrent) download(p, slot int) error {
	var s, e int
	s = (t.Torrent.NumPieces() * slot) / bucket
	/*if s < t.Torrent.NumPieces()/n {
		s = s - p

	} else if s >= t.Torrent.NumPieces()/n && s < (t.Torrent.NumPieces()*(n-1))/n {
		s = s - p/2
	}*/
	s = s - p/2
	if s < 0 {
		s = 0
	}

	if t.Torrent.NumPieces() < s+p {
		s = t.Torrent.NumPieces() - p
	}

	e = s + p

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ex := make(chan any, 1)
	t.wg.Add(1)
	go func() {
		defer t.wg.Done()
		t.Torrent.DownloadPieces(s, e)
		ex <- struct{}{}
	}()

	select {
	case <-ex:
		log.Info(ScaleBar(s, e, t.Torrent.NumPieces()), "ih", t.InfoHash(), "slot", slot, "s", s, "e", e, "p", p, "total", t.Torrent.NumPieces())
	case <-ctx.Done():
		log.Warn("Piece download timeout", "ih", t.InfoHash(), "slot", slot, "s", s, "e", e, "p", p, "total", t.Torrent.NumPieces())
		return ctx.Err()
		//case <-t.closeAll:
		//	return nil
	}

	return nil
}

func (t *Torrent) Running() bool {
	return t.status == torrentRunning
}

func (t *Torrent) Pending() bool {
	return t.status == torrentPending
}

func (t *Torrent) Stop() {
	t.Lock()
	defer t.Unlock()

	//close(t.closeAll)

	t.wg.Wait()
	t.Torrent.Drop()

	log.Info(ProgressBar(t.BytesCompleted(), t.Torrent.Length(), ""), "ih", t.InfoHash(), "total", common.StorageSize(t.Torrent.Length()), "req", common.StorageSize(t.BytesRequested()), "finish", common.StorageSize(t.Torrent.BytesCompleted()), "status", t.Status(), "cited", t.Cited())
}
