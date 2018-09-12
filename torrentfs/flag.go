package torrentfs

// Config ...
type Config struct {
	DataDir *string
	RpcURI  *string
	IpcPath *string
	DefaultTrackers *[]string
}
