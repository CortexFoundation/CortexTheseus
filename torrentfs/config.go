package torrentfs

import (
	"github.com/CortexFoundation/CortexTheseus/params"
)

// Config ...
type Config struct {
	// Host is the host interface on which to start the storage server. If this
	// field is empty, no storage will be started.
	Port            int      `toml:",omitempty"`
	DataDir         string   `toml:",omitempty"`
	RpcURI          string   `toml:",omitempty"`
	IpcPath         string   `toml:",omitempty"`
	DisableUTP      bool     `toml:",omitempty"`
	DisableDHT      bool     `toml:",omitempty"`
	DefaultTrackers []string `toml:",omitempty"`
	BoostNodes      []string `toml:",omitempty"`
	SyncMode        string   `toml:",omitempty"`
	MaxSeedingNum   int      `toml:",omitempty"`
	MaxActiveNum    int      `toml:",omitempty"`
	FullSeed        bool
}

// DefaultConfig contains default settings for the storage.
var DefaultConfig = Config{
	Port:            0,
	DefaultTrackers: params.MainnetTrackers,
	BoostNodes:      params.TorrentBoostNodes,
	SyncMode:        "full",
	DisableUTP:      false,
	DisableDHT:      false,
	MaxSeedingNum:   1024,
	MaxActiveNum:    1024,
	FullSeed:        false,
}

const (
	queryTimeInterval              = 1
	expansionFactor        float64 = 1.2
	defaultSeedInterval            = 600
	torrentWaitingTime             = 900
	downloadWaitingTime            = 1800
	defaultBytesLimitation         = 512 * 1024
	defaultTmpFilePath             = ".tmp"
	version                        = "1"
)
