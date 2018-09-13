package torrentfs

// Config ...
type Config struct {
	DataDir string
	RpcURI  string
	IpcPath string
	// Host is the host interface on which to start the storage server. If this
	// field is empty, no storage will be started.
	Host    string
	// Port is the TCP port number on which to start the storage server. The
	// default zero value is/ valid and will pick a port number randomly.
	Port    int
	DefaultTrackers []string
}

var DefaultConfig = Config{
	Host:    "localhost",
	Port:    8080,
}