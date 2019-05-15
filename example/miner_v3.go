package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"flag"
	"io"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/consensus/cuckoo"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	cuckoo_gpu "github.com/CortexFoundation/CortexTheseus/miner/cuckoocuda"
	_ "log"
	"math/rand"
	"net"
	"os"
	"time"

	// "strconv"
	"sync"
)

type Miner interface{
	Mining()
}

type Connection struct{
	lock  sync.Mutex
	state bool
}
type Cortex struct{
	server, account, chip string
	deviceId uint
	conn *net.TCPConn
	reader *bufio.Reader
	consta Connection
}

type Task struct {
	Header     string
	Nonce      string
	Solution   string
	Difficulty string
}

type ReqObj struct {
	Id      int      `json:"id"`
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

func (cm* Cortex)read() map[string]interface{} {
	rep := make([]byte, 0, 4096) // big buffer
	for {
		tmp, isPrefix, err := cm.reader.ReadLine()
		if err == io.EOF {
			fmt.Println("Tcp disconnectted")
			cm.conn.Close()
			cm.conn = nil
			cm.consta.lock.Lock()
			cm.consta.state = false
			cm.consta.lock.Unlock()
			return nil
		}
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

func (cm* Cortex)write(reqObj ReqObj) {
	req, err := json.Marshal(reqObj)
	checkError(err)

	req = append(req, uint8('\n'))
	fmt.Println("req = ", req)
	_, _ = cm.conn.Write(req)
}

//	init cortex miner
func (cm* Cortex)init() (*net.TCPConn){
	fmt.Println("init")
	//cm.server = "cortex.waterhole.xyz:8008"
	//cm.server = "localhost:8009"
	//cm.account = "0xc3d7a1ef810983847510542edfd5bc5551a6321c"
	tcpAddr, err := net.ResolveTCPAddr("tcp", cm.server)
	checkError(err)

	cm.conn, err = net.DialTCP("tcp", nil, tcpAddr)
	fmt.Println("init be")
	checkError(err)
	cm.consta.lock.Lock()
	cm.consta.state = true
	cm.consta.lock.Unlock()
	fmt.Println("there")
	cm.reader = bufio.NewReader(cm.conn)
//	defer cm.conn.Close()
	cm.conn.SetKeepAlive(true)
	fmt.Println("here", cm.consta.state, cm.conn)
	return cm.conn
}

//	miner login to mining pool
func (cm* Cortex)login() {
	fmt.Println("login")
	var reqLogin = ReqObj{
		Id:      73,
		Jsonrpc: "2.0",
		Method:  "eth_submitLogin",
		Params:  []string{cm.account},
	}
	cm.write(reqLogin)
	cm.read()
}

//	get mining task
func (cm* Cortex)getWork() {
	req := ReqObj{
	Id:      100,
	Jsonrpc: "2.0",
	Method:  "eth_getWork",
	Params:  []string{""},
	}
	cm.write(req)
}

//	submit task
func (cm* Cortex)submit(sol Task) {
	var reqSubmit = ReqObj{
		Id:      73,
		Jsonrpc: "2.0",
		Method:  "eth_submitWork",
		Params:  []string{sol.Nonce, sol.Header, sol.Solution},
	}
	cm.write(reqSubmit)
}

//	cortex mining
func (cm* Cortex) Mining() {
	for {
		for{
			cm.consta.lock.Lock()
			consta := cm.consta.state
			cm.consta.lock.Unlock()
			if consta== false {
				fmt.Println("mining")
				cm.init()
				cm.login()
			} else {
				fmt.Println("mining succeess")
				break
			}
		}
		cm.miningOnce()
	}
}

func (cm* Cortex) miningOnce() {
	fmt.Println("once")
	type TaskWrapper struct {
		Lock  sync.Mutex
		TaskQ Task
	}

	var currentTask TaskWrapper
	var taskHeader, taskNonce, taskDifficulty string
	var THREAD uint = 10
	if (cm.chip == "gpu"){
		THREAD = 1
	}
	cuckoo.CuckooInitialize(1, uint32(THREAD))
	if (cm.chip == "gpu") {
		cuckoo_gpu.CuckooInitialize(cm.deviceId)
	}
	solChan := make(chan Task, THREAD)

	for nthread := 0; nthread < int(THREAD); nthread++ {
		go func(tidx uint32, currentTask_ *TaskWrapper) {
			for {
				if cm.consta.state == false {
					return
				}
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

				if (cm.chip == "cpu") {
					status, sols := cuckoo.CuckooFindSolutions(header, curNonce)
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
				} else if (cm.chip == "gpu") {
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
			}
		}(uint32(nthread), &currentTask)
	}

	cm.getWork()
	go func(currentTask_ *TaskWrapper) {
		for {
			msg := cm.read()
			if cm.consta.state == false {
				 return
			}
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
		if cm.consta.state == false {
			return
		}
		select {
			case sol := <-solChan:
				currentTask.Lock.Lock()
				task := currentTask.TaskQ
				currentTask.Lock.Unlock()
				if sol.Header == task.Header {
					cm.submit(sol)
				}
			default:
				time.Sleep(100 * time.Millisecond)
		}
	}
	cuckoo.CuckooFinalize()
}

func init() {
	flag.BoolVar(&help, "help", false, "show help")
	flag.StringVar(&remote, "pool_uri", "localhost:8009", "romote server address")
	flag.StringVar(&account, "account", "0xc3d7a1ef810983847510542edfd5bc5551a6321c", "miner accounts")
	flag.StringVar(&chip, "devicetype", "cpu", "device type")
	flag.IntVar(&deviceId, "deviceid", 0, "gpu id to mine")
}

var help bool
var remote, account, chip string
var deviceId int

func main() {
	flag.Parse()
	if (help == true){
		fmt.Println("Usage:\ngo run miner.go -r romote -a account -c gpu\nexample:go run miner.go -r localhost:8009 -a 0xc3d7a1ef810983847510542edfd5bc5551a6321c -c cpu")
	}else{
		fmt.Println(account, remote, chip)
	}
	var cm Miner = &Cortex{
		chip: chip,
		account: account,
		server: remote,
		deviceId: uint(deviceId),
	}
	cm.Mining()
}
