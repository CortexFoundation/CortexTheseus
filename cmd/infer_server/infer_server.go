package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"sync"

	"github.com/CortexFoundation/CortexTheseus/inference"
	"github.com/CortexFoundation/CortexTheseus/inference/synapse"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rpc"
)

var (
	storageDir = flag.String("storage_dir", "~/.cortex/warehouse", "Inference server's data dir, absolute path")
	logLevel   = flag.Int("verbosity", 3, "Log level to emit to screen")
	port       = flag.Int("port", 8827, "Server listen port")
	IsNotCache = flag.Bool("disable_cache", false, "Disable cache")
	DeviceType = flag.String("device", "cpu", "cpu or gpu")
)

var rpcClient *rpc.Client
var simpleCache sync.Map

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

    log.Trace("Handler Info", "request", r)

	switch inference.RetriveType(body) {
	case inference.INFER_BY_IH:
		var iw inference.IHWork
		if err := json.Unmarshal(body, &iw); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		infoHashHandler(w, &iw)
		break

	case inference.INFER_BY_IC:
		var iw inference.ICWork
		if err := json.Unmarshal(body, &iw); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		inputContentHandler(w, &iw)
		break

	case inference.GAS_BY_H:
		var iw inference.GasWork
		if err := json.Unmarshal(body, &iw); err != nil {
			RespErrorText(w, ErrDataParse)
		}
		gasHandler(w, &iw)
		break


	default:
		RespErrorText(w, ErrInvalidInferTaskType)
		break
	}
}

func main() {
	flag.Parse()

	// Set log
	log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(*logLevel), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))

	log.Info("Inference Server", "Help Command", "./infer_server -h")

	inferServer := synapse.New(synapse.Config{
		StorageDir: *storageDir,
		IsNotCache: *IsNotCache,
		DeviceType: *DeviceType,
	})
	log.Info("Initilized inference server with synapse engine")

	http.HandleFunc("/", handler)

	log.Info(fmt.Sprintf("Http Server Listen on 0.0.0.0:%v", *port))
	err := http.ListenAndServe(fmt.Sprintf(":%v", *port), nil)

	log.Error(fmt.Sprintf("Server Closed with Error %v", err))
	inferServer.Close()
}
