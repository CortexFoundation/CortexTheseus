/*
  This file is part of go-ethereum

  go-ethereum is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  go-ethereum is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with go-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
package rpcws

import (
	"fmt"
	"net"
	"net/http"

	"code.google.com/p/go.net/websocket"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/rpc"
	"github.com/ethereum/go-ethereum/xeth"
)

var wslogger = logger.NewLogger("RPC-WS")
var JSON rpc.JsonWrapper

type WebSocketServer struct {
	pipe     *xeth.XEth
	port     int
	doneCh   chan bool
	listener net.Listener
}

func NewWebSocketServer(pipe *xeth.XEth, port int) (*WebSocketServer, error) {
	sport := fmt.Sprintf(":%d", port)
	l, err := net.Listen("tcp", sport)
	if err != nil {
		return nil, err
	}

	return &WebSocketServer{
		pipe,
		port,
		make(chan bool),
		l,
	}, nil
}

func (self *WebSocketServer) handlerLoop() {
	for {
		select {
		case <-self.doneCh:
			wslogger.Infoln("Shutdown RPC-WS server")
			return
		}
	}
}

func (self *WebSocketServer) Stop() {
	close(self.doneCh)
}

func (self *WebSocketServer) Start() {
	wslogger.Infof("Starting RPC-WS server on port %d", self.port)
	go self.handlerLoop()

	api := rpc.NewEthereumApi(self.pipe)
	h := self.apiHandler(api)
	http.Handle("/ws", h)

	err := http.Serve(self.listener, nil)
	if err != nil {
		wslogger.Errorln("Error on RPC-WS interface:", err)
	}
}

func (s *WebSocketServer) apiHandler(api *rpc.EthereumApi) http.Handler {
	fn := func(w http.ResponseWriter, req *http.Request) {
		h := sockHandler(api)
		s := websocket.Server{Handler: h}
		s.ServeHTTP(w, req)
	}

	return http.HandlerFunc(fn)
}

func sockHandler(api *rpc.EthereumApi) websocket.Handler {
	var jsonrpcver string = "2.0"
	fn := func(conn *websocket.Conn) {
		for {
			wslogger.Debugln("Handling connection")
			var reqParsed rpc.RpcRequest

			// reqParsed, reqerr := JSON.ParseRequestBody(conn.Request())
			if err := websocket.JSON.Receive(conn, &reqParsed); err != nil {
				jsonerr := &rpc.RpcErrorObject{-32700, rpc.ErrorParseRequest}
				JSON.Send(conn, &rpc.RpcErrorResponse{JsonRpc: jsonrpcver, ID: nil, Error: jsonerr})
				continue
			}

			var response interface{}
			reserr := api.GetRequestReply(&reqParsed, &response)
			if reserr != nil {
				wslogger.Warnln(reserr)
				jsonerr := &rpc.RpcErrorObject{-32603, reserr.Error()}
				JSON.Send(conn, &rpc.RpcErrorResponse{JsonRpc: jsonrpcver, ID: reqParsed.ID, Error: jsonerr})
				continue
			}

			wslogger.Debugf("Generated response: %T %s", response, response)
			JSON.Send(conn, &rpc.RpcSuccessResponse{JsonRpc: jsonrpcver, ID: reqParsed.ID, Result: response})
		}
	}
	return websocket.Handler(fn)
}
