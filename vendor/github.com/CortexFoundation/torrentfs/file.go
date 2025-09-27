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
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/metainfo"
	cp "github.com/otiai10/copy"

	"github.com/CortexFoundation/torrentfs/backend"
	"github.com/CortexFoundation/torrentfs/backend/caffe"
	"github.com/CortexFoundation/torrentfs/params"
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
					ex, co, to, e := fs.storage().ExistsOrActive(ctx, infohash, rawSize)
					log.Warn("Timeout", "ih", infohash, "size", common.StorageSize(rawSize), "err", ctx.Err(), "retry", fs.retry.Load(), "complete", common.StorageSize(co), "timeout", to, "exist", ex, "err", e)
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
			fs.encounter(infohash)
		}
		return ret, nil
	}
}

// Seeding Local File, validate folder, seeding and
// load files, default mode is copyMode, linkMode
// will limit user's operations for original files
func (fs *TorrentFS) SeedingLocal(ctx context.Context, filePath string, isLinkMode bool) (infoHash string, err error) {
	// 1. Check root folder
	if _, err = os.Stat(filePath); err != nil {
		return "", fmt.Errorf("stat filePath failed: %w", err)
	}

	// 2. Ensure there is at least one non-empty file
	var hasValidFile func(basePath string, fi os.FileInfo) bool
	hasValidFile = func(basePath string, fi os.FileInfo) bool {
		path := filepath.Join(basePath, fi.Name())
		if fi.IsDir() {
			dirFp, e := os.Open(path)
			if e != nil {
				return false
			}
			fInfos, e := dirFp.Readdir(0)
			dirFp.Close()
			if e != nil {
				log.Error("Read dir failed", "filePath", path, "err", e)
				return false
			}
			for _, v := range fInfos {
				if hasValidFile(path, v) {
					return true
				}
			}
			return false
		}
		return fi.Size() > 0
	}

	var (
		dataInfo os.FileInfo
		fileMode bool
	)
	dataPath := filepath.Join(filePath, "data")
	if dataInfo, err = os.Stat(dataPath); err != nil {
		dataPath = filePath
		if dataInfo, err = os.Stat(dataPath); err != nil {
			return "", fmt.Errorf("load data failed: %w", err)
		}
		fileMode = true
	}

	if !hasValidFile(filePath, dataInfo) {
		return "", fmt.Errorf("seedingLocal: empty seeding data, path=%s fileMode=%t", dataPath, fileMode)
	}

	// 3. Build torrent metainfo
	mi := metainfo.MetaInfo{
		AnnounceList: [][]string{params.MainnetTrackers},
	}
	mi.SetDefaults()

	info := metainfo.Info{PieceLength: 256 * 1024}
	if err = info.BuildFromFilePath(dataPath); err != nil {
		return "", fmt.Errorf("build metainfo failed: %w", err)
	}
	if mi.InfoBytes, err = bencode.Marshal(info); err != nil {
		return "", fmt.Errorf("marshal metainfo failed: %w", err)
	}

	torrentPath := filepath.Join(filePath, "torrent")
	if fileMode {
		torrentPath = "torrent"
	}

	// safer: buffer + atomic write
	var buf bytes.Buffer
	if err = mi.Write(&buf); err != nil {
		return "", fmt.Errorf("encode metainfo failed: %w", err)
	}
	if err = os.WriteFile(torrentPath, buf.Bytes(), 0644); err != nil {
		return "", fmt.Errorf("write torrent file failed: %w", err)
	}

	// 4. Copy or symlink
	ih := common.Address(mi.HashInfoBytes())
	infoHash = strings.TrimPrefix(strings.ToLower(ih.Hex()), common.Prefix)
	linkDst := filepath.Join(fs.storage().TmpDataDir(), infoHash)

	log.Info("Local file Seeding", "ih", infoHash, "path", dataPath, "len", info.Length)

	if !isLinkMode {
		if !fileMode {
			err = cp.Copy(filePath, linkDst)
		} else {
			if err = os.MkdirAll(filepath.Dir(linkDst), 0777); err != nil {
				return "", fmt.Errorf("mkdir failed: %w", err)
			}
			if err = cp.Copy(filePath, filepath.Join(linkDst, dataInfo.Name())); err != nil {
				return "", fmt.Errorf("copy data failed: %w", err)
			}
			if err = cp.Copy(torrentPath, filepath.Join(linkDst, "torrent")); err != nil {
				return "", fmt.Errorf("copy torrent failed: %w", err)
			}
		}
	} else {
		if fileMode {
			return "", fmt.Errorf("link mode not supported in fileMode=true")
		}
		if _, err = os.Stat(linkDst); err == nil {
			err = os.ErrExist
		} else {
			var absOriFilePath string
			if absOriFilePath, err = filepath.Abs(filePath); err == nil {
				err = os.Symlink(absOriFilePath, linkDst)
			}
		}
	}

	// 5. Start seeding
	if err == nil || errors.Is(err, os.ErrExist) {
		log.Debug("SeedingLocal", "dest", linkDst, "err", err)
		if e := fs.storage().Search(ctx, infoHash, 1024*1024*1024); e == nil {
			fs.storage().AddLocalSeedFile(infoHash)
		} else {
			return "", fmt.Errorf("search storage failed: %w", e)
		}
	}

	return infoHash, err
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
