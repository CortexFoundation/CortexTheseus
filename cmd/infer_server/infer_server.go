package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"

	infer "github.com/ethereum/go-ethereum/inference/synapse"
	"github.com/ethereum/go-ethereum/log"
	"github.com/ethereum/go-ethereum/rpc"
)

var (
	storageDir = flag.String("storageDir", "/home/wlt/InferenceServer/warehouse", "Inference server's data dir, absolute path")
	cortexURI  = flag.String("cortexURI", "http://localhost:25667", "Cortex core binary's rpc")
	logLevel   = flag.Int("logLevel", 3, "Log level to emit to screen")
	port       = flag.Int("port", 8827, "Server listen port")
	IsNotCache = flag.Bool("disable_cache", false, "Disable cache")
)

const (
	INFER_TASK_BY_INFO_HASH     = 1
	INFER_TASK_BY_INPUT_CONTENT = 2
)

var rpcClient *rpc.Client

type InferWork struct {
	// default is zero
	Type uint32

	ModelHash string
	InputHash string

	InputAddress string
	InputSlot    string
}

func handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		RespErrorText(w, ErrRequestMethodNotPost)
		return
	}

	body, rerr := ioutil.ReadAll(r.Body)
	if rerr != nil {
		RespErrorText(w, ErrRequestBodyRead)
		return
	}

	log.Trace("Handler Info", "request", r, "body", string(body))

	var inferWork InferWork

	if err := json.Unmarshal(body, &inferWork); err != nil {
		RespErrorText(w, ErrDataParse)
		return
	}

	switch inferWork.Type {
	case INFER_TASK_BY_INFO_HASH:
		infoHashHandler(w, &inferWork)
	case INFER_TASK_BY_INPUT_CONTENT:
		inputContentHandler(w, &inferWork)
	default:
		defaultHandler(w, &inferWork)
	}

}

func main() {
	flag.Parse()

	// Set log
	log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(*logLevel), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))

	log.Info("Inference Server", "Help Command", "./infer_server -h")

	var err error
	if rpcClient, err = rpc.DialContext(context.Background(), *cortexURI); err != nil {
		log.Error("Cortex core RPC dial", "error", err)
		return
	}
	log.Info("Initilized RPC client", "cortex uri", *cortexURI)

	inferServer := infer.New(infer.Config{
		StorageDir: *storageDir,
		IsNotCache: *IsNotCache,
	})
	log.Info("Initilized inference server with synapse engine")

	http.HandleFunc("/", handler)

	log.Info(fmt.Sprintf("Http Server Listen on 0.0.0.0:%v", *port))
	err = http.ListenAndServe(fmt.Sprintf(":%v", *port), nil)

	log.Error(fmt.Sprintf("Server Closed with Error %v", err))
	inferServer.Close()
}
