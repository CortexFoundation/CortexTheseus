package torrentfs

// Config ...
type Config struct {
	DataDir string `toml:",omitempty"`
	RpcURI  string `toml:",omitempty"`
	IpcPath string `toml:",omitempty"`
	// Host is the host interface on which to start the storage server. If this
	// field is empty, no storage will be started.
	Host    string `toml:",omitempty"`
	// Port is the TCP port number on which to start the storage server. The
	// default zero value is/ valid and will pick a port number randomly.
	Port    int `toml:",omitempty"`
	DefaultTrackers []string `toml:",omitempty"`
}
// DefaultConfig contains default settings for the storage.
var DefaultConfig = Config{
	Host:    "localhost",
	Port:    8085,
	DefaultTrackers: []string{"http://47.52.39.170:5008/announce"},
}