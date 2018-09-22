package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus/cuckoo"
	"github.com/ethereum/go-ethereum/core/types"
	cuckoo_gpu "github.com/ethereum/go-ethereum/miner/cuckoocuda"
	"log"
	"math/rand"
	"net"
	"os"
	"time"

	// "strconv"
	"sync"
	"flag"
)

type Task struct {
	Header     string
	Nonce      string
	Solution   string
	Difficulty string
}

type ReqObj struct {
	Id      int      `json:"id"` // struct标签， 如果指定，jsonrpc包会在序列化json时，将该聚合字段命名为指定的字符串
	Jsonrpc string   `json:"jsonrpc"`
	Method  string   `json:"method"`
	Params  []string `json:"params"`
}

func checkError(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "Fatal error: %s", err.Error())
		os.Exit(1)
	}
}

func read(reader *bufio.Reader) map[string]interface{} {
	rep := make([]byte, 0, 4096) // big buffer
	for {
		tmp, isPrefix, err := reader.ReadLine()
		checkError(err)
		rep = append(rep, tmp...)
		if isPrefix == false {
			break
		}
	}
	// fmt.Println("received ", len(rep), " bytes: ", string(rep), "\n")
	var repObj map[string]interface{}
	err := json.Unmarshal(rep, &repObj)
	checkError(err)
	return repObj
}

func write(reqObj ReqObj, conn *net.TCPConn) {
	req, err := json.Marshal(reqObj)
	checkError(err)

	req = append(req, uint8('\n'))
	_, _ = conn.Write(req)
}

func main() {
	deviceId := flag.Int("deviceid", 0, "gpu id to mine")
	flag.Parse()

	var testFixTask bool = false
	if testFixTask {
		fmt.Println("testFixTask = ", testFixTask)

	}

	type TaskWrapper struct {
		Lock  sync.Mutex
		TaskQ Task
	}
	var currentTask TaskWrapper

	var THREAD uint = 1
	cuckoo.CuckooInitialize(1, uint32(THREAD))
	cuckoo_gpu.CuckooInitialize(uint(*deviceId))
	log.Println("Device ID: ", *deviceId)
	var taskHeader, taskNonce, taskDifficulty string
	//-------- connect to server -------------
	// var server = "139.196.32.192:8009"
	// var server = "cortex.waterhole.xyz:8008"
	// var server = "192.168.50.104:8009"
	var server = "localhost:8009"
	tcpAddr, err := net.ResolveTCPAddr("tcp", server)
	checkError(err)
	conn, err := net.DialTCP("tcp", nil, tcpAddr)
	checkError(err)
	reader := bufio.NewReader(conn)
	defer conn.Close()
	var reqLogin = ReqObj{
		Id:      73,
		Jsonrpc: "2.0",
		Method:  "eth_submitLogin",
		Params:  []string{"0xc3d7a1ef810983847510542edfd5bc5551a6321c"},
	}
	write(reqLogin, conn)
	loginRsp := read(reader)
	fmt.Println("loginRsp: ", loginRsp)

	solChan := make(chan Task, THREAD)
	for nthread := 0; nthread < int(THREAD); nthread++ {
		go func(tidx uint32, currentTask_ *TaskWrapper) {
			for {
				currentTask_.Lock.Lock()
				task := currentTask_.TaskQ
				currentTask_.Lock.Unlock()
				if len(task.Difficulty) == 0 {
					time.Sleep(100 * time.Millisecond)
					continue
				}
				tgtDiff := common.HexToHash(task.Difficulty[2:])
				header, _ := hex.DecodeString(task.Header[2:])
				var result types.BlockSolution
				curNonce := uint64(rand.Int63())
				// fmt.Println("task: ", header[:], curNonce)
				status, sols := cuckoo_gpu.CuckooFindSolutionsCuda(header, curNonce)
				if status != 0 {
					fmt.Println("result: ", status, sols)
					for _, solUint32 := range sols {
						var sol types.BlockSolution
						copy(sol[:], solUint32)
						sha3hash := common.BytesToHash(cuckoo.Sha3Solution(&sol))
						fmt.Println(curNonce, "\n sol hash: ", hex.EncodeToString(sha3hash.Bytes()), "\n tgt hash: ", hex.EncodeToString(tgtDiff.Bytes()))
						if sha3hash.Big().Cmp(tgtDiff.Big()) <= 0 {
							result = sol
							nonceStr := common.Uint64ToHexString(uint64(curNonce))
							digest := common.Uint32ArrayToHexString([]uint32(result[:]))
							ok, _ := cuckoo.CuckooVerifyHeaderNonceSolutionsDifficulty(header[:], curNonce, &sol)
							if !ok {
								fmt.Println("verify failed", header[:], curNonce, &sol)
							} else {
								solChan <- Task{Nonce: nonceStr, Header: taskHeader, Solution: digest}
							}
						}
					}
				}
			}
		}(uint32(nthread), &currentTask)
	}

	write(ReqObj{
		Id:      100,
		Jsonrpc: "2.0",
		Method:  "eth_getWork",
		Params:  []string{""},
	}, conn)

	go func(currentTask_ *TaskWrapper) {
		for {
			msg := read(reader)
			fmt.Println("Received: ", msg)
			reqId, _ := msg["id"].(float64)
			if uint32(reqId) == 100 || uint32(reqId) == 0 {
				workInfo, _ := msg["result"].([]interface{})
				if len(workInfo) >= 3 {
					taskHeader, taskNonce, taskDifficulty = workInfo[0].(string), workInfo[1].(string), workInfo[2].(string)
					fmt.Println("Get Work: ", taskHeader, taskNonce, taskDifficulty)
					currentTask_.Lock.Lock()
					currentTask_.TaskQ.Nonce = taskNonce
					currentTask_.TaskQ.Header = taskHeader
					currentTask_.TaskQ.Difficulty = taskDifficulty
					currentTask_.Lock.Unlock()
				}
			}
		}
	}(&currentTask)
	time.Sleep(2 * time.Second)
	for {
		select {
		case sol := <-solChan:
			currentTask.Lock.Lock()
			task := currentTask.TaskQ
			currentTask.Lock.Unlock()
			if sol.Header == task.Header {
				var reqSubmit = ReqObj{
					Id:      73,
					Jsonrpc: "2.0",
					Method:  "eth_submitWork",
					Params:  []string{sol.Nonce, sol.Header, sol.Solution},
				}
				write(reqSubmit, conn)
			}
		default:
			time.Sleep(100 * time.Millisecond)
		}
	}
	cuckoo.CuckooFinalize()
}
