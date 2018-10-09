package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"

	infer "github.com/ethereum/go-ethereum/infer_server"
	"github.com/ethereum/go-ethereum/log"
)

var (
	storageDir = flag.String("storageDir", "/home/wlt/InferenceServer/warehouse", "Inference server's data dir, absolute path")
	logLevel   = flag.Int("logLevel", 3, "Log level to emit to screen")
	port       = flag.Int("port", 8827, "Server listen port")
	IsNotCache = flag.Bool("disable_cache", false, "Disable cache")
)

type InferWork struct {
	ModelHash string
	InputHash string
}

func LocalInfer(modelHash, inputHash string) (uint64, error) {
	var (
		resultCh = make(chan uint64, 1)
		errCh    = make(chan error, 1)
	)

	err := infer.SubmitInferWork(
		modelHash,
		inputHash,
		resultCh,
		errCh)

	if err != nil {
		return 0, err
	}

	select {
	case result := <-resultCh:
		return result, nil
	case err := <-errCh:
		return 0, err
	}

	return 0, nil
}

func handler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		fmt.Fprintf(w, `{"msg": "error", "info": "Request method not POST"}`)
		return
	}

	body, rerr := ioutil.ReadAll(r.Body)
	if rerr != nil {
		fmt.Fprintf(w, `{"msg": "error", "info": "Read request body error"}`)
		return
	}

	log.Trace("Handler Info", "request", r, "body", string(body))

	var inferWork InferWork

	if err := json.Unmarshal(body, &inferWork); err != nil {
		fmt.Fprintf(w, `{"msg": "error", "info": "Data parse error"}`)
		return
	}

	if inferWork.ModelHash == "" || inferWork.InputHash == "" {
		fmt.Fprintf(w, `{"msg": "error", "info": "Data is empty"}`)
		return
	}

	log.Info("Infer Work", "Model Hash", inferWork.ModelHash, "Input Hash", inferWork.InputHash)

	label, err := LocalInfer(inferWork.ModelHash, inferWork.InputHash)
	log.Info("Infer Result", "label", label, "error", err)

	if err != nil {
		fmt.Fprintf(w, fmt.Sprintf(`{"msg": "error", "info": "%v"}`, err))
		return
	}

	fmt.Fprintf(w, fmt.Sprintf(`{"msg": "ok", "info": "%v"}`, label))
}

func main() {
	// Set log
	log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(*logLevel), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))
	log.Info("Inference Server", "Help Command", "./infer_server -h")

	flag.Parse()

	inferServer := infer.New(infer.Config{
		StorageDir: *storageDir,
		IsNotCache: *IsNotCache,
	})

	http.HandleFunc("/", handler)

	log.Info(fmt.Sprintf("Http Server Listen on 0.0.0.0:%v", *port))
	err := http.ListenAndServe(fmt.Sprintf(":%v", *port), nil)

	log.Error(fmt.Sprintf("Server Closed with Error %v", err))
	inferServer.Close()
}
