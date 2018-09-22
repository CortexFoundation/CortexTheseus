package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"

	infer "github.com/ethereum/go-ethereum/infernet"
	"github.com/ethereum/go-ethereum/log"
)

var (
	forcePending = flag.Bool("forcePending", false, "Force program wait for file sync done")
	storageDir   = flag.String("storageDir", "~/InferenceServer/warehouse", "Set the inference server's data dir")
	logLevel     = flag.Int("logLevel", 3, "Log level to emit to screen")
)

type InferWork struct {
	ModelHash string
	InputHash string
}

func LocalInfer(modelHash, inputHash string, forcePending bool) (uint64, error) {
	var (
		resultCh = make(chan uint64, 1)
		errCh    = make(chan error, 1)
	)

	err := infer.SubmitInferWork(
		modelHash,
		inputHash,
		forcePending,
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

	fmt.Println("Accept request ", r)

	body, rerr := ioutil.ReadAll(r.Body)
	if rerr != nil {
		fmt.Fprintf(w, `{"msg": "error", "info": "Read request body error"}`)
		return
	}

	var inferWork InferWork

	if err := json.Unmarshal(body, &inferWork); err != nil {
		fmt.Fprintf(w, `{"msg": "error": "info": "Data parse error"}`)
		return
	}

	if inferWork.ModelHash == "" || inferWork.InputHash == "" {
		fmt.Fprintf(w, `{"msg": "error": "info": "Data parse error"}`)
		return
	}

	label, err := LocalInfer(inferWork.ModelHash, inferWork.InputHash, *forcePending)

	if err != nil {
		fmt.Fprintf(w, `{"msg": "error": "info": "Infer core error"}`)
		return
	}

	fmt.Fprintf(w, fmt.Sprintf(`{"msg": "ok": "info": "%v"}`, label))
}

func main() {
	flag.Parse()

	// Set log
	log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(*logLevel), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))

	inferServer := infer.New(infer.Config{
		StorageDir: *storageDir,
	})

	http.HandleFunc("/", handler)
	err := http.ListenAndServe(":8827", nil)

	log.Error(fmt.Sprintf("Server Closed with Error %v", err))
	inferServer.Close()
}
