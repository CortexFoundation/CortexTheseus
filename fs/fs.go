package fs

import (
  "math"
  "torrent/libtorrent"
)

func Host(torrentFiles []string) (err error)  {
  flags := &torrent.TorrentFlags{
    Dial:                nil,
    Port:                7777,
    FileDir:             ".",
    SeedRatio:           math.Inf(0),
    UseDeadlockDetector: true,
    UseLPD:              true,
    UseDHT:              true,
    UseUPnP:             false,
    UseNATPMP:           false,
    TrackerlessMode:     false,
                        // IP address of gateway
    Gateway:            "",
    InitialCheck:       true,
    FileSystemProvider: torrent.OsFsProvider{},
    Cacher:             torrent.NewRamCacheProvider(2048),
    ExecOnSeeding:      "",
    QuickResume:        true,
    MaxActive:          16,
    MemoryPerTorrent:   -1,
  }
  torrent.RunTorrents(flags, torrentFiles)
  return
}
