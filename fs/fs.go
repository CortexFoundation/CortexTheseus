package fs

import (
	"math"
	torrent "torrent/libtorrent"

	_ "../libtorrent"
)

func Host(torrentFiles []string) (err error) {
	StartListening()

	return
	flags := &torrent.TorrentFlags{
		Dial:                nil,
		Port:                7777,
		FileDir:             "/home/lizhen/storage",
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
		MaxActive:          128,
		MemoryPerTorrent:   -1,
	}
	torrent.RunTorrents(flags, torrentFiles)
	return
}
