package cortexminer

import (
	"bufio"
	"sync"
	"net"
	"github.com/CortexFoundation/CortexTheseus/solution/config"
)

type Miner interface {
	Mining()
}

type Connection struct {
	lock  sync.Mutex
	state bool
}

type Cortex struct {
	server, account string
	deviceInfos       []config.DeviceInfo
	conn            *net.TCPConn
	reader          *bufio.Reader
	consta          Connection
	share_accepted  int
	share_rejected  int
	param config.Param
}

type ReqObj struct {
	Id      int      `json:"id"`
	Jsonrpc string   `json:"jsonrpc"`
	Method  string   `json:"method"`
	Params  []string `json:"params"`
}



func (cortex Cortex) New(deviceInfos []config.DeviceInfo, param config.Param) Cortex{
	cortex.param = param
	cortex.deviceInfos = deviceInfos
	cortex.share_accepted = 0
	cortex.share_rejected = 0
	return cortex
}


