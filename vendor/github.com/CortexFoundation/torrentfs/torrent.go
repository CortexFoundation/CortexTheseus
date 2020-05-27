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
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"os"
	"path"
	"path/filepath"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
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
		err := os.MkdirAll(filepath.Dir(path.Join(t.filepath, "data")), 0750) //os.ModePerm)
		if err != nil {
			return
		}
	}
	//log.Info("Try to boost files", "files", files)
	for i, filename := range files {
		filePath := path.Join(t.filepath, filename)
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
	mi, err := metainfo.LoadFromFile(path.Join(t.filepath, "torrent"))
	if err != nil {
		log.Error("Error while loading torrent", "Err", err)
		return
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	spec.Storage = storage.NewFile(t.filepath)
	//spec.Trackers = append(spec.Trackers, tm.trackers...)
	//spec.Trackers = tm.trackers
	if torrent, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		//var ss []string
		//slices.MakeInto(&ss, mi.Nodes)
		//tm.client.AddDHTNodes(ss)
		//<-torrent.GotInfo()
		//torrent.VerifyData()
		t.Torrent = torrent
		//	t.Pause()
	}
}

func (t *Torrent) ReloadTorrent(data []byte, tm *TorrentManager) error {
	err := os.Remove(path.Join(t.filepath, ".torrent.bolt.db"))
	if err != nil {
		log.Warn("Remove path failed", "path", path.Join(t.filepath, ".torrent.bolt.db"), "err", err)
	}
	/*torrentPath := path.Join(t.filepath, "torrent")
	f, err := os.Create(torrentPath)
	if err != nil {
		log.Warn("Create torrent path failed", "path", torrentPath)
		return
	}
	defer f.Close()
	log.Debug("Write seed file (Boost mode)", "path", torrentPath)
	if _, err := f.Write(data); err != nil {
		log.Error("Error while write torrent file", "error", err)
		return
	}*/
	buf := bytes.NewBuffer(data)
	mi, err := metainfo.Load(buf)

	//mi, err := metainfo.LoadFromFile(torrentPath)
	if err != nil {
		log.Error("Error while adding torrent", "Err", err)
		return err
	}
	spec := torrent.TorrentSpecFromMetaInfo(mi)
	spec.Storage = storage.NewFile(t.filepath)
	//spec.Trackers = append(spec.Trackers, tm.trackers...)
	//spec.Trackers = tm.trackers
	spec.Trackers = nil
	if torrent, _, err := tm.client.AddTorrentSpec(spec); err == nil {
		//var ss []string
		//slices.MakeInto(&ss, mi.Nodes)
		//tm.client.AddDHTNodes(ss)
		//<-torrent.GotInfo()
		//torrent.VerifyData()
		t.Torrent = torrent
		//tm.torrents[spec.InfoHash] = t
		//t.Pause()
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
	return t.Seeding()
}

func (t *Torrent) WriteTorrent() error {
	if _, err := os.Stat(path.Join(t.filepath, "torrent")); err == nil {
		t.Pause()
		return nil
	}

	if f, err := os.OpenFile(path.Join(t.filepath, "torrent"), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0660); err == nil {
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

func (t *Torrent) Seed() {
	if t.Torrent.Info() == nil || t.status == torrentSeeding {
		return
	}
	//t.status = torrentSeeding
	//if t.currentConns == 0 {
	//	t.currentConns = t.maxEstablishedConns
	//	t.Torrent.SetMaxEstablishedConns(t.currentConns)
	//}

	//t.Torrent.DownloadAll()
	if t.currentConns <= t.minEstablishedConns {
		t.currentConns = t.maxEstablishedConns
		t.Torrent.SetMaxEstablishedConns(t.currentConns)
		//if t.currentConns < t.minEstablishedConns {
		//	t.currentConns = t.minEstablishedConns
	}
	//t.Torrent.SetMaxEstablishedConns(t.currentConns)
	//t.Torrent.SetMaxEstablishedConns(0)
	//}
	if t.Torrent.Seeding() {
		t.status = torrentSeeding
		elapsed := time.Duration(mclock.Now()) - time.Duration(t.start)
		log.Info("Imported new segment", "hash", common.HexToHash(t.InfoHash()), "size", common.StorageSize(t.BytesCompleted()), "files", len(t.Files()), "pieces", t.Torrent.NumPieces(), "seg", len(t.Torrent.PieceStateRuns()), "cited", t.cited, "peers", t.currentConns, "status", t.status, "elapsed", common.PrettyDuration(elapsed))
		//t.Torrent.Drop()
	}
}

func (t *Torrent) Seeding() bool {
	return t.Torrent.Info() != nil && t.status == torrentSeeding && t.BytesMissing() == 0
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
		//t.Torrent.Drop()
	}
}

func (t *Torrent) Paused() bool {
	return t.status == torrentPaused
}

//func (t *Torrent) Length() int64 {
//	return t.bytesCompleted + t.bytesMissing
//}

//func (t *Torrent) NumPieces() int {
//	return t.Torrent.NumPieces()
//}

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
	//log.Info("Limit mode", "hash", t.infohash, "fast", t.fast, "conn", t.currentConns, "request", t.bytesRequested, "limit", limitPieces, "cur", t.maxPieces, "total", t.Torrent.NumPieces())
	t.status = torrentRunning
	if limitPieces != t.maxPieces {
		t.maxPieces = limitPieces
		//t.Torrent.DownloadPieces(0, limitPieces)
		t.download(limitPieces, slot)
	}
}

func (t *Torrent) download(p, slot int) {
	//if p >= t.Torrent.NumPieces() {
	//	t.Torrent.DownloadAll()
	//	return
	//}

	var s, e int
	/*if mod == 0 {
		e = p
	} else if mod == 1 {
		s = (t.Torrent.NumPieces() - p) / 2
		e = (t.Torrent.NumPieces() + p) / 2
	} else if mod == 2 {
		if  t.Torrent.NumPieces() < mod {
			s = mod - t.Torrent.NumPieces()
		}
		if t.Torrent.NumPieces() < mod + p {
			s = t.Torrent.NumPieces() - p
		}
		s = mod
		e = s + p
	} else {
		s = t.Torrent.NumPieces() - p
		e = t.Torrent.NumPieces()
	}*/
	s = (t.Torrent.NumPieces() * slot) / bucket
	if s < t.Torrent.NumPieces()/3 {
		s = s - p

	} else if s >= t.Torrent.NumPieces()/3 && s < (t.Torrent.NumPieces()*2)/3 {
		s = s - p/2
	}

	if s < 0 {
		s = 0
	}

	if t.Torrent.NumPieces() < s+p {
		s = t.Torrent.NumPieces() - p
	}

	e = s + p
	//log.Trace("[ "+progress+" ]", "hash", t.infohash, "b", s, "e", e, "p", p, "t", t.Torrent.NumPieces(), "s", slot, "b", bucket, "conn", t.currentConns)
	t.Torrent.DownloadPieces(s, e)
}

func (t *Torrent) Running() bool {
	return t.status == torrentRunning
}

func (t *Torrent) Finished() bool {
	//for _, file := range t.Files() {
	//	if file.BytesCompleted() <= 0 {
	//		return false
	//	}
	//log.Info("File", "hash", t.InfoHash(), "name", file.Path(), "complete", file.BytesCompleted(), "file", file)
	//}
	return t.bytesMissing == 0 && t.bytesRequested > 0 && t.bytesCompleted > 0
}

func (t *Torrent) Pending() bool {
	return t.status == torrentPending
}
