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
	"fmt"
	"strings"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"

	"github.com/CortexFoundation/torrentfs/backend/caffe"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
)

// can only call by fs.go: 'SeedingLocal()'
func (tm *TorrentManager) AddLocalSeedFile(ih string) bool {
	if !common.IsHexAddress(ih) {
		return false
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	if _, ok := params.GoodFiles[ih]; ok {
		return false
	}

	tm.localSeedLock.Lock()
	tm.localSeedFiles[ih] = true
	tm.localSeedLock.Unlock()

	return true
}

// only files in map:localSeedFile can be paused!
func (tm *TorrentManager) PauseLocalSeedFile(ih string) error {
	if !common.IsHexAddress(ih) {
		return errors.New("invalid infohash format")
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	tm.localSeedLock.Lock()
	defer tm.localSeedLock.Unlock()

	if valid, ok := tm.localSeedFiles[ih]; !ok {
		return errors.New(fmt.Sprintf("Not Local Seeding File<%s>", ih))
	} else if _, ok := params.GoodFiles[ih]; ok {
		return errors.New(fmt.Sprintf("Cannot Pause On-Chain GoodFile<%s>", ih))
	} else if !valid {
		return errors.New(fmt.Sprintf("Local Seeding File Is Not Seeding<%s>", ih))
	}

	if t := tm.getTorrent(ih); t != nil {
		log.Debug("TorrentFS", "from seed to pause", "ok")
		t.Pause()
		tm.localSeedFiles[ih] = !t.Paused()
	}

	return nil
}

// only files in map:localSeedFile can be resumed!
func (tm *TorrentManager) ResumeLocalSeedFile(ih string) error {
	if !common.IsHexAddress(ih) {
		return errors.New("invalid infohash format")
	}
	ih = strings.TrimPrefix(strings.ToLower(ih), common.Prefix)

	tm.localSeedLock.Lock()
	defer tm.localSeedLock.Unlock()

	if valid, ok := tm.localSeedFiles[ih]; !ok {
		return errors.New(fmt.Sprintf("Not Local Seeding File<%s>", ih))
	} else if _, ok := params.GoodFiles[ih]; ok {
		return errors.New(fmt.Sprintf("Cannot Operate On-Chain GoodFile<%s>", ih))
	} else if valid {
		return errors.New(fmt.Sprintf("Local Seeding File Is Already Seeding<%s>", ih))
	}

	if t := tm.getTorrent(ih); t != nil {
		resumeFlag := t.Seed()
		log.Debug("TorrentFS", "from pause to seed", resumeFlag)
		tm.localSeedFiles[ih] = resumeFlag
	}

	return nil
}

// divide localSeed/on-chain Files
// return status of torrents
func (tm *TorrentManager) ListAllTorrents() map[string]map[string]int {
	tm.lock.RLock()
	tm.localSeedLock.RLock()
	defer tm.lock.RUnlock()
	defer tm.localSeedLock.RUnlock()

	tts := make(map[string]map[string]int, tm.torrents.Len())
	/*for ih, tt := range tm.torrents {
		tType := torrentTypeOnChain
		if _, ok := tm.localSeedFiles[ih]; ok {
			tType = torrentTypeLocal
		}
		tts[ih] = map[string]int{
			"status": tt.Status(),
			"type":   tType,
		}
	}*/

	tm.torrents.Range(func(ih string, tt *caffe.Torrent) bool {
		tType := torrentTypeOnChain
		if _, ok := tm.localSeedFiles[ih]; ok {
			tType = torrentTypeLocal
		}
		if tt.Status() < 5 {
			tts[ih] = map[string]int{
				"status":  tt.Status(),
				"type":    tType,
				"period":  int(tt.Cited()),
				"size":    int(tt.Length()),
				"pending": int(tt.BytesRequested()),
			}
		}
		return true
	})

	return tts
}

func (tm *TorrentManager) Metrics() time.Duration {
	return tm.updates
}

func (tm *TorrentManager) LocalPort() int {
	return tm.client.LocalPort()
}

func (tm *TorrentManager) Congress() int {
	return int(tm.seeds.Load()) //tm.seedingTorrents.Len()
}

func (tm *TorrentManager) Candidate() int {
	return int(tm.actives.Load())
}

func (tm *TorrentManager) Nominee() int {
	//return tm.pendingTorrents.Len()
	return int(tm.pends.Load())
}

func (tm *TorrentManager) IsPending(ih string) bool {
	//return tm.pendingTorrents[ih] != nil
	//_, ok := tm.pendingTorrents.Get(ih)
	//return ok
	//	return tm.pendingTorrents.Has(ih)
	if t := tm.getTorrent(ih); t != nil {
		return t.Status() == caffe.TorrentPending
	}
	return false
}

func (tm *TorrentManager) IsDownloading(ih string) bool {
	//return tm.activeTorrents[ih] != nil
	//_, ok := tm.activeTorrents.Get(ih)
	//return ok
	//return tm.activeTorrents.Has(ih)
	if t := tm.getTorrent(ih); t != nil {
		return t.Status() == caffe.TorrentRunning
	}
	return false
}

func (tm *TorrentManager) IsSeeding(ih string) bool {
	//return tm.seedingTorrents[ih] != nil
	//_, ok := tm.seedingTorrents.Get(ih)
	//return ok
	if t := tm.getTorrent(ih); t != nil {
		return t.Status() == caffe.TorrentSeeding
	}
	return false //tm.seedingTorrents.Has(ih)
}

/*func (tm *TorrentManager) GlobalTrackers() [][]string {
	tm.lock.RLock()
	defer tm.lock.RUnlock()

	return tm.globalTrackers
}*/

// Search and donwload files from torrent
func (tm *TorrentManager) Search(ctx context.Context, hex string, request uint64) error {
	if !common.IsHexAddress(hex) {
		return errors.New("invalid infohash format")
	}

	hex = strings.TrimPrefix(strings.ToLower(hex), common.Prefix)

	if params.IsBad(hex) {
		return nil
	}

	if request == 0x7fffffffffffffff {
		// TODO 0x7fffffffffffffff local downloading file
		// GoodFiles[hex] = false
	}

	//if tm.mode == params.FULL {
	//if request == 0 {
	//      log.Warn("Prepare mode", "ih", hex)
	//      request = uint64(block)
	//}
	//}

	downloadMeter.Mark(1)

	if request == 0 {
		// sync create torrent
		//	tm.addInfoHash(hex, int64(request))
		//	return nil
	}

	//return tm.commit(ctx, hex, request)
	return tm.taskEvent.Post(mainEvent{B: types.NewBitsFlow(hex, request)})
}

// Add torrent to the leeching loop
//func (tm *TorrentManager) commit(ctx context.Context, hex string, request uint64) error {
/*select {
case tm.taskChan <- types.NewBitsFlow(hex, request):
case <-ctx.Done():
	return ctx.Err()
case <-tm.closeAll:
}

return nil */

//	return tm.taskEvent.Post(mainEvent{types.NewBitsFlow(hex, request)})
//}
