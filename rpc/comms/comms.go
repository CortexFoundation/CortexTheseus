package comms

import (
	"io"
	"net"

	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/logger/glog"
	"github.com/ethereum/go-ethereum/rpc/api"
	"github.com/ethereum/go-ethereum/rpc/codec"
	"github.com/ethereum/go-ethereum/rpc/shared"
	"strings"
)

const (
	maxHttpSizeReqLength = 1024 * 1024 // 1MB
)

var (
	// List with all API's which are offered over the in proc interface by default
	DefaultInProcApis = api.AllApis

	// List with all API's which are offered over the IPC interface by default
	DefaultIpcApis = api.AllApis

	// List with API's which are offered over thr HTTP/RPC interface by default
	DefaultHttpRpcApis = strings.Join([]string{
		api.DbApiName, api.EthApiName, api.NetApiName, api.Web3ApiName,
		}, ",")
)

type EthereumClient interface {
	// Close underlaying connection
	Close()
	// Send request
	Send(interface{}) error
	// Receive response
	Recv() (interface{}, error)
}

func handle(conn net.Conn, api api.EthereumApi, c codec.Codec) {
	codec := c.New(conn)

	for {
		req, err := codec.ReadRequest()
		if err == io.EOF {
			codec.Close()
			return
		} else if err != nil {
			glog.V(logger.Error).Infof("comms recv err - %v\n", err)
			codec.Close()
			return
		}

		var rpcResponse interface{}
		res, err := api.Execute(req)

		rpcResponse = shared.NewRpcResponse(req.Id, req.Jsonrpc, res, err)
		err = codec.WriteResponse(rpcResponse)
		if err != nil {
			glog.V(logger.Error).Infof("comms send err - %v\n", err)
			codec.Close()
			return
		}
	}
}
