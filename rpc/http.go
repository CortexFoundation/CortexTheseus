package rpc

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"

	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/xeth"
	"github.com/rs/cors"
)

var rpclogger = logger.NewLogger("RPC")

const (
	jsonrpcver       = "2.0"
	maxSizeReqLength = 1024 * 1024 // 1MB
)

func Start(pipe *xeth.XEth, config RpcConfig) error {
	l, err := net.Listen("tcp", fmt.Sprintf("%s:%d", config.ListenAddress, config.ListenPort))
	if err != nil {
		rpclogger.Errorf("Can't listen on %s:%d: %v", config.ListenAddress, config.ListenPort, err)
		return err
	}

	var handler http.Handler
	if len(config.CorsDomain) > 0 {
		var opts cors.Options
		opts.AllowedMethods = []string{"POST"}
		opts.AllowedOrigins = []string{config.CorsDomain}

		c := cors.New(opts)
		handler = c.Handler(JSONRPC(pipe))
	} else {
		handler = JSONRPC(pipe)
	}

	go http.Serve(l, handler)

	return nil
}

// JSONRPC returns a handler that implements the Ethereum JSON-RPC API.
func JSONRPC(pipe *xeth.XEth) http.Handler {
	api := NewEthereumApi(pipe)

	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		// Limit request size to resist DoS
		if req.ContentLength > maxSizeReqLength {
			jsonerr := &RpcErrorObject{-32700, "Request too large"}
			send(w, &RpcErrorResponse{Jsonrpc: jsonrpcver, Id: nil, Error: jsonerr})
			return
		}

		// Read request body
		defer req.Body.Close()
		body, err := ioutil.ReadAll(req.Body)
		if err != nil {
			jsonerr := &RpcErrorObject{-32700, "Could not read request body"}
			send(w, &RpcErrorResponse{Jsonrpc: jsonrpcver, Id: nil, Error: jsonerr})
		}

		// Try to parse the request as a single
		var reqSingle RpcRequest
		if err := json.Unmarshal(body, &reqSingle); err == nil {
			response := RpcResponse(api, &reqSingle)
			send(w, &response)
			return
		}

		// Try to parse the request to batch
		var reqBatch []RpcRequest
		if err := json.Unmarshal(body, &reqBatch); err == nil {
			// Build response batch
			resBatch := make([]*interface{}, len(reqBatch))
			for i, request := range reqBatch {
				response := RpcResponse(api, &request)
				resBatch[i] = response
			}
			send(w, resBatch)
			return
		}

		// Not a batch or single request, error
		jsonerr := &RpcErrorObject{-32600, "Could not decode request"}
		send(w, &RpcErrorResponse{Jsonrpc: jsonrpcver, Id: nil, Error: jsonerr})
	})
}

func RpcResponse(api *EthereumApi, request *RpcRequest) *interface{} {
	var reply, response interface{}
	reserr := api.GetRequestReply(request, &reply)
	switch reserr.(type) {
	case nil:
		response = &RpcSuccessResponse{Jsonrpc: jsonrpcver, Id: request.Id, Result: reply}
	case *NotImplementedError:
		jsonerr := &RpcErrorObject{-32601, reserr.Error()}
		response = &RpcErrorResponse{Jsonrpc: jsonrpcver, Id: request.Id, Error: jsonerr}
	case *DecodeParamError, *InsufficientParamsError, *ValidationError, *InvalidTypeError:
		jsonerr := &RpcErrorObject{-32602, reserr.Error()}
		response = &RpcErrorResponse{Jsonrpc: jsonrpcver, Id: request.Id, Error: jsonerr}
	default:
		jsonerr := &RpcErrorObject{-32603, reserr.Error()}
		response = &RpcErrorResponse{Jsonrpc: jsonrpcver, Id: request.Id, Error: jsonerr}
	}

	rpclogger.DebugDetailf("Generated response: %T %s", response, response)
	return &response
}

func send(writer io.Writer, v interface{}) (n int, err error) {
	var payload []byte
	payload, err = json.MarshalIndent(v, "", "\t")
	if err != nil {
		rpclogger.Fatalln("Error marshalling JSON", err)
		return 0, err
	}
	rpclogger.DebugDetailf("Sending payload: %s", payload)

	return writer.Write(payload)
}
