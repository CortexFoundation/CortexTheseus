package torrentfs

import (
	"github.com/CortexFoundation/CortexTheseus/params"
)

// Config ...
type Config struct {
	// Host is the host interface on which to start the storage server. If this
	// field is empty, no storage will be started.
<<<<<<< HEAD
	Port                int       `toml:",omitempty"`
	DataDir             string    `toml:",omitempty"`
	RpcURI              string    `toml:",omitempty"`
	IpcPath             string    `toml:",omitempty"`
	DisableUTP          bool      `toml:",omitempty"`
	DefaultTrackers     []string  `toml:",omitempty"`
	BoostNodes          []string  `toml:",omitempty"`
	SyncMode            string    `toml:",omitempty"`
	MaxSeedingNum       int       `toml:",omitempty"`
	MaxActiveNum        int       `toml:",omitempty"`
=======
	Host string `toml:",omitempty"`
	// Port is the TCP port number on which to start the storage server. The
	// default zero value is/ valid and will pick a port number randomly.
	Port            int    `toml:",omitempty"`
	DisableUTP      bool   `toml:",omitempty"`
	DefaultTrackers []string `toml:",omitempty"`
	SyncMode        string `toml:",omitempty"`
	TestMode        bool   `toml:",omitempty"`
>>>>>>> parent of 5579c49... stash
}

// DefaultConfig contains default settings for the storage.
var DefaultConfig = Config{
<<<<<<< HEAD
	Port: 30090,
	DefaultTrackers: params.MainnetTrackers,
	BoostNodes: params.TorrentBoostNodes,
=======
	Host:            "localhost",
	Port:            8085,
	DefaultTrackers: params.MainnetTrackers, //"http://torrent.cortexlabs.ai:5008/announce",
	//DefaultTrackers: "http://47.52.39.170:5008/announce",
>>>>>>> parent of 5579c49... stash
	SyncMode: "full",
	TestMode: false,
	DisableUTP: true,
	MaxSeedingNum: 640,
	MaxActiveNum: 128,
}
