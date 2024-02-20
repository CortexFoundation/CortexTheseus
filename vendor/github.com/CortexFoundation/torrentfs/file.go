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
	"errors"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	//"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/backend"
	"github.com/CortexFoundation/torrentfs/backend/caffe"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/metainfo"
	cp "github.com/otiai10/copy"
)

func (fs *TorrentFS) GetFileWithSize(ctx context.Context, infohash string, rawSize uint64, subpath string) ([]byte, error) {
	log.Debug("Get file with size", "ih", infohash, "size", common.StorageSize(rawSize), "path", subpath)
	if ret, mux, err := fs.storage().GetFile(ctx, infohash, subpath); err != nil {
		fs.wg.Add(1)
		go func(ctx context.Context, ih string) {
			defer fs.wg.Done()
			fs.wakeup(ctx, ih)
		}(ctx, infohash)

		if params.IsGood(infohash) {
			//start := mclock.Now()
			//log.Info("Downloading ... ...", "ih", infohash, "size", common.StorageSize(rawSize), "neighbors", fs.Neighbors(), "current", fs.monitor.CurrentNumber())

			if mux != nil {
				sub := mux.Subscribe(caffe.TorrentEvent{})
				defer sub.Unsubscribe()

				select {
				case <-sub.Chan():
					//log.Info("Seeding notify !!! !!!", "ih", infohash, "size", common.StorageSize(rawSize), "neighbors", fs.Neighbors(), "current", fs.monitor.CurrentNumber(), "ev", ev.Data)
					if ret, _, err := fs.storage().GetFile(ctx, infohash, subpath); err != nil {
						log.Debug("File downloading ... ...", "ih", infohash, "size", common.StorageSize(rawSize), "path", subpath, "err", err)
					} else {
						//elapsed := time.Duration(mclock.Now()) - time.Duration(start)
						//log.Info("Downloaded", "ih", infohash, "size", common.StorageSize(rawSize), "neighbors", fs.Neighbors(), "elapsed", common.PrettyDuration(elapsed), "current", fs.monitor.CurrentNumber())
						if uint64(len(ret)) > rawSize {
							return nil, backend.ErrInvalidRawSize
						}
						return ret, err
					}
				case <-ctx.Done():
					fs.retry.Add(1)
					ex, co, to, _ := fs.storage().ExistsOrActive(ctx, infohash, rawSize)
					log.Warn("Timeout", "ih", infohash, "size", common.StorageSize(rawSize), "err", ctx.Err(), "retry", fs.retry.Load(), "complete", common.StorageSize(co), "timeout", to, "exist", ex)
					return nil, ctx.Err()
				case <-fs.closeAll:
					log.Info("File out")
					return nil, nil
				}
			} else {
				t := time.NewTimer(1000 * time.Millisecond)
				defer t.Stop()
				for {
					select {
					case <-t.C:
						if ret, _, err := fs.storage().GetFile(ctx, infohash, subpath); err != nil {
							log.Debug("File downloading ... ...", "ih", infohash, "size", common.StorageSize(rawSize), "path", subpath, "err", err)
							t.Reset(1000 * time.Millisecond)
						} else {
							//elapsed := time.Duration(mclock.Now()) - time.Duration(start)
							//log.Info("Downloaded", "ih", infohash, "size", common.StorageSize(rawSize), "neighbors", fs.Neighbors(), "elapsed", common.PrettyDuration(elapsed), "current", fs.monitor.CurrentNumber())
							if uint64(len(ret)) > rawSize {
								return nil, backend.ErrInvalidRawSize
							}
							return ret, err
						}
					case <-ctx.Done():
						fs.retry.Add(1)
						ex, co, to, _ := fs.storage().ExistsOrActive(ctx, infohash, rawSize)
						log.Warn("Timeout", "ih", infohash, "size", common.StorageSize(rawSize), "err", ctx.Err(), "retry", fs.retry.Load(), "complete", common.StorageSize(co), "timeout", to, "exist", ex)
						return nil, ctx.Err()
					case <-fs.closeAll:
						log.Info("File out")
						return nil, nil
					}
				}
			}
		}

		return nil, err
	} else {
		if uint64(len(ret)) > rawSize {
			return nil, backend.ErrInvalidRawSize
		}
		log.Debug("Get File directly", "ih", infohash, "size", common.StorageSize(rawSize), "path", subpath, "ret", len(ret))
		if !params.IsGood(infohash) {
			go fs.encounter(infohash)
		}
		return ret, nil
	}
}

// Seeding Local File, validate folder, seeding and
// load files, default mode is copyMode, linkMode
// will limit user's operations for original files
func (fs *TorrentFS) SeedingLocal(ctx context.Context, filePath string, isLinkMode bool) (infoHash string, err error) {
	// 1. check folder exist
	if _, err = os.Stat(filePath); err != nil {
		return
	}

	// 2. check subfile data exist and not empty:
	// recursively iterate until meet file not empty
	var iterateForValidFile func(basePath string, dataInfo os.FileInfo) bool
	iterateForValidFile = func(basePath string, dataInfo os.FileInfo) bool {
		filePath := filepath.Join(basePath, dataInfo.Name())
		if dataInfo.IsDir() {
			dirFp, _ := os.Open(filePath)
			if fInfos, err := dirFp.Readdir(0); err != nil {
				log.Error("Read dir failed", "filePath", filePath, "err", err)
				return false
			} else {
				for _, v := range fInfos {
					// return as soon as possible if meet 'true', else continue
					if iterateForValidFile(filePath, v) {
						return true
					}
				}
			}
		} else if dataInfo.Size() > 0 {
			return true
		}
		return false
	}

	var (
		dataInfo os.FileInfo
		fileMode bool = false
	)
	dataPath := filepath.Join(filePath, "data")
	if dataInfo, err = os.Stat(dataPath); err != nil {
		dataPath = filepath.Join(filePath, "")
		if dataInfo, err = os.Stat(dataPath); err != nil {
			log.Error("Load data failed", "dataPath", dataPath)
			return
		}
		fileMode = true
	}
	validFlag := iterateForValidFile(filePath, dataInfo)
	if !validFlag {
		err = errors.New("SeedingLocal: Empty Seeding Data!")
		log.Error("SeedingLocal", "check", err.Error(), "path", dataPath, "name", dataInfo.Name(), "fileMode", fileMode)
		return
	}

	// 3. generate torrent file, rewrite if exists
	mi := metainfo.MetaInfo{
		AnnounceList: [][]string{params.MainnetTrackers},
	}
	mi.SetDefaults()
	info := metainfo.Info{PieceLength: 256 * 1024}
	if err = info.BuildFromFilePath(dataPath); err != nil {
		return
	}
	if mi.InfoBytes, err = bencode.Marshal(info); err != nil {
		return
	}

	torrentPath := filepath.Join(filePath, "torrent")
	if fileMode {
		torrentPath = filepath.Join("", "torrent")
	}

	var fileTorrent *os.File
	fileTorrent, err = os.OpenFile(torrentPath, os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	if err = mi.Write(fileTorrent); err != nil {
		return
	}

	// 4. copy or link, will not cover if dst exist!
	ih := common.Address(mi.HashInfoBytes())
	log.Info("Local file Seeding", "ih", ih.Hex(), "path", dataPath)
	infoHash = strings.TrimPrefix(strings.ToLower(ih.Hex()), common.Prefix)
	linkDst := filepath.Join(fs.storage().TmpDataDir, infoHash)
	if !isLinkMode {
		if !fileMode {
			err = cp.Copy(filePath, linkDst)
		} else {
			err = os.MkdirAll(filepath.Dir(linkDst), 0777) //os.FileMode(os.ModePerm))
			if err != nil {
				log.Error("Mkdir failed", "path", linkDst)
				return
			}

			err = cp.Copy(filePath, filepath.Join(linkDst, dataInfo.Name()))
			if err != nil {
				log.Error("Mkdir failed", "filePath", filePath, "path", linkDst)
				return
			}
			log.Info("Torrent copy", "torrentPath", torrentPath, "linkDst", linkDst)
			err = cp.Copy(torrentPath, filepath.Join(linkDst, "torrent"))
			if err != nil {
				log.Error("Mkdir failed", "torrentPath", torrentPath, "path", linkDst)
				return
			}

		}
	} else {
		if fileMode {
			//TODO
			log.Error("Not support", "link", isLinkMode)
			return
		}
		// check if symbol link exist
		if _, err = os.Stat(linkDst); err == nil {
			// choice-1: original symbol link exists, cover it. (passed)

			// choice-2: original symbol link exists, return error
			err = os.ErrExist
		} else {
			// create symbol link
			var absOriFilePath string
			if absOriFilePath, err = filepath.Abs(filePath); err == nil {
				err = os.Symlink(absOriFilePath, linkDst)
			}
		}
	}

	// 5. seeding
	if err == nil || errors.Is(err, os.ErrExist) {
		log.Debug("SeedingLocal", "dest", linkDst, "err", err)
		err = fs.storage().Search(ctx, ih.Hex(), 0)
		if err == nil {
			fs.storage().AddLocalSeedFile(infoHash)
		}
	}

	return
}

// PauseSeeding Local File
func (fs *TorrentFS) PauseLocalSeed(ctx context.Context, ih string) error {
	return fs.storage().PauseLocalSeedFile(ih)
}

// ResumeSeeding Local File
func (fs *TorrentFS) ResumeLocalSeed(ctx context.Context, ih string) error {
	return fs.storage().ResumeLocalSeedFile(ih)
}

// List All Torrents Status (read-only)
func (fs *TorrentFS) ListAllTorrents(ctx context.Context) map[string]map[string]int {
	return fs.storage().ListAllTorrents()
}

func (fs *TorrentFS) Tunnel(ctx context.Context, ih string) error {
	if err := fs.storage().Search(ctx, ih, 1024*1024*1024); err != nil {
		return err
	}
	return nil
}

func (fs *TorrentFS) Drop(ih string) error {
	if err := fs.storage().Dropping(ih); err != nil {
		return err
	}
	return nil
}

func (fs *TorrentFS) Download(ctx context.Context, ih string, request uint64) error {
	if request > 0 {
		return fs.bitsflow(ctx, ih, request)
	}
	return fs.download(ctx, ih, request)
}

func (fs *TorrentFS) Status(ctx context.Context, ih string) (int, error) {
	if fs.storage().IsPending(ih) {
		return STATUS_PENDING, nil
	}

	if fs.storage().IsDownloading(ih) {
		return STATUS_RUNNING, nil
	}

	if fs.storage().IsSeeding(ih) {
		return STATUS_SEEDING, nil
	}

	return STATUS_UNKNOWN, nil
}

func (fs *TorrentFS) LocalPort() int {
	return fs.storage().LocalPort()
}

func (fs *TorrentFS) Congress() int {
	return fs.storage().Congress()
}

func (fs *TorrentFS) Candidate() int {
	return fs.storage().Candidate()
}

func (fs *TorrentFS) NasCounter() uint64 {
	return fs.nasCounter
}

func (fs *TorrentFS) Nominee() int {
	return fs.storage().Nominee()
}

func (fs *TorrentFS) IsActive(err error) bool {
	return !errors.Is(err, backend.ErrInactiveTorrent)
}
