package main

import (
	"flag"
	"os"
	"strings"


	"github.com/CortexFoundation/CortexTheseus/torrentfs/types"
	download "github.com/CortexFoundation/CortexTheseus/torrentfs/manager"
	"github.com/CortexFoundation/CortexTheseus/torrentfs/monitor"
)

func main() {
	os.Exit(mainExitCode())
}

func mainExitCode() int {
	// DataDir := "/data/serving/InferenceServer/warehouse"
	DataDir := flag.String("d", "/home/lizhen/storage", "storage path")
	RpcURI := flag.String("r", "http://192.168.5.11:28888", "json-rpc uri")
	IpcPath := flag.String("i", "", "ipc socket path")
	trackerURI := flag.String("t", "http://47.52.39.170:5008/announce", "tracker uri")
	flag.Parse()

	trackers := strings.Split(*trackerURI, ",")
	f := &types.Flag{
		DataDir,
		RpcURI,
		IpcPath,
		&trackers,
	}

	dlCilent := download.NewTorrentManager(f)
	m := monitor.NewMonitor(f)
	m.SetDownloader(dlCilent)
	m.Start()
	return 0
}
