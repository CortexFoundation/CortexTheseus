package torrentfs

import (
	"github.com/CortexFoundation/CortexTheseus/params"
)

// Config ...
type Config struct {
	DataDir string `toml:",omitempty"`
	RpcURI  string `toml:",omitempty"`
	IpcPath string `toml:",omitempty"`
	// Host is the host interface on which to start the storage server. If this
	// field is empty, no storage will be started.
	DisableUTP      bool   `toml:",omitempty"`
	DefaultTrackers []string `toml:",omitempty"`
	SyncMode        string `toml:",omitempty"`
}

// DefaultConfig contains default settings for the storage.
var DefaultConfig = Config{
	DefaultTrackers: params.MainnetTrackers, //"http://torrent.cortexlabs.ai:5008/announce",
	//DefaultTrackers: "http://47.52.39.170:5008/announce",
	SyncMode: "full",
	DisableUTP: true,
}
