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

package torrentfs

import (
	"bytes"
	"os"
	"path/filepath"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/anacrolix/torrent"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/anacrolix/torrent/storage"
)

type Torrent struct {
	*torrent.Torrent
	maxEstablishedConns int
	minEstablishedConns int
	currentConns        int
	bytesRequested      int64
	bytesLimitation     int64
	bytesCompleted      int64
	bytesMissing        int64
	status              int
	infohash            string
	filepath            string
	cited               int64
	weight              int
	loop                int
	maxPieces           int
	isBoosting          bool
	fast                bool
	start               mclock.AbsTime
	ch                  chan bool
}

func (t *Torrent) BytesLeft() int64 {
	if t.bytesRequested < t.bytesCompleted {
		return 0
	}
	return t.bytesRequested - t.bytesCompleted
}

func (t *Torrent) InfoHash() string {
	return t.infohash
}

func (t *Torrent) ReloadFile(files []string, datas [][]byte, tm *TorrentManager) {
	if len(files) > 1 {
		err := os.MkdirAll(filepath.Dir(filepath.Join(t.filepath, "data")), 0600) //os.ModePerm)
		if err != nil {
			return
		}
	}
	for i, filename := range files {
		filePath := filepath.Join(t.filepath, filename)
		f, err := os.OpenFile(filePath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0660)
		if err != nil {
			return
		}
		defer f.Close()
		log.Debug("Write file (Boost mode)", "path", filePath)
		if _, err := f.Write(datas[i]); err != nil {
			log.Error("Error while write data file", "error", err)
		}
	}
	mi, err := metainfo.LoadFromFile(filepath.Join(t.filepath, "torrent"))
	if err != nil {
		log.Error("Error while loading torrent", "Err", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	spec.Storage = storage.NewFile(t.filepath)
	if torrent, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		t.Torrent = torrent
	}
}

func (t *Torrent) ReloadTorrent(data []byte, tm *TorrentManager) error {
	//err := os.Remove(filepath.Join(t.filepath, ".torrent.bolt.db"))
	//if err != nil {
	//	log.Warn("Remove path failed", "path", filepath.Join(t.filepath, ".torrent.bolt.db"), "err", err)
	//}

	buf := bytes.NewBuffer(data)
	mi, err := metainfo.Load(buf)

	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return err
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	//spec.Storage = storage.NewFile(t.filepath)
	//spec.Trackers = nil
	//t.Drop()
	if torrent, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		t.Torrent = torrent
	} else {
		return err
	}
	return nil
}

func (t *Torrent) Ready() bool {
	if _, ok := BadFiles[t.InfoHash()]; ok {
		return false
	}
	t.cited += 1
	ret := t.IsSeeding()
	if !ret {
		log.Debug("Not ready", "ih", t.InfoHash(), "status", t.status, "seed", t.Torrent.Seeding(), "seeding", torrentSeeding)
	}

	return ret
}

func (t *Torrent) WriteTorrent() error {
	if _, err := os.Stat(filepath.Join(t.filepath, "torrent")); err == nil {
		t.Pause()
		return nil
	}

	if f, err := os.OpenFile(filepath.Join(t.filepath, "torrent"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0660); err == nil {
		defer f.Close()
		log.Debug("Write seed file", "path", t.filepath)
		if err := t.Metainfo().Write(f); err == nil {
			t.Pause()
			return f.Close()
		} else {
			log.Warn("Write seed error", "err", err)
			return err
		}
	} else {
		log.Warn("Create Path error", "err", err)
		return err
	}
}

func (t *Torrent) BoostOff() {
	t.isBoosting = false
}

func (t *Torrent) Seed() bool {
	if t.Torrent.Info() == nil {
		log.Debug("Torrent info is nil", "ih", t.InfoHash())
		return false
	}
	if t.status == torrentSeeding {
		log.Debug("Torrent status is", "status", t.status, "ih", t.InfoHash())
		return true
	}
	if t.currentConns <= t.minEstablishedConns {
		t.currentConns = t.maxEstablishedConns
		t.Torrent.SetMaxEstablishedConns(t.currentConns)
	}
	if t.Torrent.Seeding() {
		t.status = torrentSeeding
		elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
		if active, ok := GoodFiles[t.InfoHash()]; !ok {
			log.Warn("New active nas found", "ih", t.InfoHash(), "ok", ok, "active", active, "size", common.StorageSize(t.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "cited", t.cited, "peers", t.currentConns, "status", t.status, "elapsed", common.PrettyDuration(elapsed))
		} else {
			log.Info("Imported new nas segment", "ih", t.InfoHash(), "size", common.StorageSize(t.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "cited", t.cited, "peers", t.currentConns, "status", t.status, "elapsed", common.PrettyDuration(elapsed))
		}
		return true
	}

	return false
}

func (t *Torrent) IsSeeding() bool {
	return t.status == torrentSeeding && t.Torrent.Seeding()
}

func (t *Torrent) Pause() {
	if t.currentConns > t.minEstablishedConns {
		t.currentConns = t.minEstablishedConns
		t.Torrent.SetMaxEstablishedConns(t.minEstablishedConns)
	}
	if t.status != torrentPaused {
		t.status = torrentPaused
		t.maxPieces = 0 //t.minEstablishedConns
		t.Torrent.CancelPieces(0, t.Torrent.NumPieces())
	}
}

func (t *Torrent) Paused() bool {
	return t.status == torrentPaused
}

func (t *Torrent) Run(slot int) {
	limitPieces := int((t.bytesRequested*int64(t.Torrent.NumPieces()) + t.Length() - 1) / t.Length())
	if limitPieces > t.Torrent.NumPieces() {
		limitPieces = t.Torrent.NumPieces()
	}

	if limitPieces <= t.maxPieces && t.status == torrentRunning {
		return
	}

	if t.fast {
		if t.currentConns <= t.minEstablishedConns {
			t.currentConns = t.maxEstablishedConns
			t.Torrent.SetMaxEstablishedConns(t.currentConns)
		}
	} else {
		if t.currentConns > t.minEstablishedConns {
			t.currentConns = t.minEstablishedConns
			t.Torrent.SetMaxEstablishedConns(t.currentConns)
		}
	}
	t.status = torrentRunning
	if limitPieces != t.maxPieces {
		t.maxPieces = limitPieces
		t.download(limitPieces, slot)
	}
}

// Find out the start and end
func (t *Torrent) download(p, slot int) {
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
	log.Info("Donwloaded pieces", "ih", t.Torrent.InfoHash(), "s", s, "e", e, "p", p, "total", t.Torrent.NumPieces())
	t.Torrent.DownloadPieces(s, e)
}

func (t *Torrent) Running() bool {
	return t.status == torrentRunning
}

func (t *Torrent) Finished() bool {
	return t.bytesMissing == 0 && t.bytesRequested > 0 && t.bytesCompleted > 0
}

func (t *Torrent) Pending() bool {
	return t.status == torrentPending
}
