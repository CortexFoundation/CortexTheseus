package torrentfs

// Flag ...
type Flag struct {
	DataDir *string
	RpcURI  *string
	IpcPath *string
	DefaultTrackers *[]string
}
