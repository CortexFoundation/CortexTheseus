package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
//	"os"
	"time"

	"github.com/PoolMiner/common"
	"github.com/PoolMiner/crypto"
	"github.com/PoolMiner/miner/libcuckoo"
	"github.com/PoolMiner/verify"

	"sync"
	"strings"
	"strconv"
)

type Miner interface {
	Mining()
}

type Connection struct {
	lock  sync.Mutex
	state bool
}
type DeviceId struct {
	lock sync.Mutex
	deviceId uint32
	use_time int64
	solution_count int64
}

type Cortex struct {
	server, account        string
	deviceIds		[]DeviceId
	verboseLevel uint
	conn                   *net.TCPConn
	reader                 *bufio.Reader
	consta                 Connection
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

func checkError(err error, func_name string) {
	if err != nil {
		log.Println(func_name, err.Error())
//		os.Exit(1)
	}
}

func (cm *Cortex) read() map[string]interface{} {
	rep := make([]byte, 0, 4096) // big buffer
	for {
		tmp, isPrefix, err := cm.reader.ReadLine()
		if err == io.EOF {
			log.Println("Tcp disconnectted")
			cm.conn.Close()
			cm.conn = nil
			cm.consta.lock.Lock()
			cm.consta.state = false
			cm.consta.lock.Unlock()
			return nil
		}
		checkError(err, "read()")
		rep = append(rep, tmp...)
		if isPrefix == false {
			break
		}
	}
	// fmt.Println("received ", len(rep), " bytes: ", string(rep), "\n")
	var repObj map[string]interface{}
	err := json.Unmarshal(rep, &repObj)
	checkError(err, "read()")
	return repObj
}

func (cm *Cortex) write(reqObj ReqObj) {
	req, err := json.Marshal(reqObj)
	checkError(err, "write()")

	req = append(req, uint8('\n'))
	_, _ = cm.conn.Write(req)
}

//	init cortex miner
func (cm *Cortex) init() *net.TCPConn {
	log.Println("Cortex Init")
	//cm.server = "cortex.waterhole.xyz:8008"
	//cm.server = "localhost:8009"
	//cm.account = "0xc3d7a1ef810983847510542edfd5bc5551a6321c"
	tcpAddr, err := net.ResolveTCPAddr("tcp", cm.server)
	checkError(err, "init()")

	cm.conn, err = net.DialTCP("tcp", nil, tcpAddr)
	checkError(err, "init()")
	cm.consta.lock.Lock()
	cm.consta.state = true
	cm.consta.lock.Unlock()
	cm.reader = bufio.NewReader(cm.conn)
	cm.conn.SetKeepAlive(true)
	return cm.conn
}

//	miner login to mining pool
func (cm *Cortex) login() {
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
func (cm *Cortex) getWork() {
	req := ReqObj{
		Id:      100,
		Jsonrpc: "2.0",
		Method:  "eth_getWork",
		Params:  []string{""},
	}
	cm.write(req)
}

//	submit task
func (cm *Cortex) submit(sol Task) {
	var reqSubmit = ReqObj{
		Id:      73,
		Jsonrpc: "2.0",
		Method:  "eth_submitWork",
		Params:  []string{sol.Nonce, sol.Header, sol.Solution},
	}
	cm.write(reqSubmit)
}

//	cortex mining
func (cm *Cortex) Mining() {
	var iDeviceIds []uint32
	for i := 0; i < len(cm.deviceIds); i++{
		iDeviceIds = append(iDeviceIds, cm.deviceIds[i].deviceId)
	}
	libcuckoo.CuckooInitialize(iDeviceIds, (uint32)(len(iDeviceIds)))

	for {
		for {
			cm.consta.lock.Lock()
			consta := cm.consta.state
			cm.consta.lock.Unlock()
			if consta == false {
				cm.init()
				cm.login()
			} else {
				break
			}
		}
		cm.miningOnce()
	}
}

func (cm *Cortex) miningOnce() {
	type TaskWrapper struct {
		Lock  sync.Mutex
		TaskQ Task
	}

	var currentTask TaskWrapper
	var taskHeader, taskNonce, taskDifficulty string
	var THREAD uint = (uint)(len(cm.deviceIds))
	rand.Seed(time.Now().UTC().UnixNano())
	solChan := make(chan Task, THREAD)
	for nthread := 0; nthread < int(THREAD); nthread++ {
		go func(tidx uint32, currentTask_ *TaskWrapper) {
			var start_time int64 = time.Now().UnixNano() / 1e6
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
				var result common.BlockSolution
				curNonce := uint64(rand.Int63())
				// fmt.Println("task: ", header[:], curNonce)
				cm.deviceIds[tidx].lock.Lock()
				status, sols := libcuckoo.FindSolutionsByGPU(header, curNonce, tidx)
				cm.deviceIds[tidx].lock.Unlock()
				if status != 0 {
					if verboseLevel >= 3 {
						log.Println("result: ", status, sols)
					}
					for _, solUint32 := range sols {
						var sol common.BlockSolution
						copy(sol[:], solUint32)
						sha3hash := common.BytesToHash(crypto.Sha3Solution(&sol))
						if verboseLevel >= 3 {
							log.Println(curNonce, "\n sol hash: ", hex.EncodeToString(sha3hash.Bytes()), "\n tgt hash: ", hex.EncodeToString(tgtDiff.Bytes()))
						}
						if sha3hash.Big().Cmp(tgtDiff.Big()) <= 0 {
							log.Println("Target Difficulty satisfied")
							result = sol
							nonceStr := common.Uint64ToHexString(uint64(curNonce))
							digest := common.Uint32ArrayToHexString([]uint32(result[:]))
							ok := verify.CuckooVerifyProof(header[:], curNonce, &sol[0], 12, 28)
							if ok != 1 {
								log.Println("verify failed", header[:], curNonce, &sol)
							} else {
								log.Println("verify successed", header[:], curNonce, &sol)
								solChan <- Task{Nonce: nonceStr, Header: taskHeader, Solution: digest}
								end_time := time.Now().UnixNano() / 1e6
								cm.deviceIds[tidx].use_time += (end_time - start_time)
								cm.deviceIds[tidx].solution_count += 1
								log.Println(fmt.Sprintf("thread %v: solutions=%v, all_time = %vms, avg_time = %vms", tidx, cm.deviceIds[tidx].solution_count, cm.deviceIds[tidx].use_time, (cm.deviceIds[tidx].use_time)/(cm.deviceIds[tidx].solution_count)))
								start_time = end_time
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
			if cm.verboseLevel >= 4 {
				log.Println("Received: ", msg)
			}
			reqId, _ := msg["id"].(float64)
			if uint32(reqId) == 100 || uint32(reqId) == 0 {
				workInfo, _ := msg["result"].([]interface{})
				if len(workInfo) >= 3 {
					taskHeader, taskNonce, taskDifficulty = workInfo[0].(string), workInfo[1].(string), workInfo[2].(string)
					log.Println("Get Work: ", taskHeader, taskDifficulty)
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
}

func init() {
	flag.BoolVar(&help, "help", false, "show help")
	flag.StringVar(&remote, "pool_uri", "miner-cn.cortexlabs.ai:8009", "mining pool address")
	flag.StringVar(&account, "account", "0xc3d7a1ef810983847510542edfd5bc5551a6321c", "miner accounts")
	flag.StringVar(&strDeviceId, "deviceids", "0", "which GPU device use for mining")
	flag.IntVar(&verboseLevel, "verbosity", 0, "verbosity level")
}

var help bool
var remote, account string
var strDeviceId string
var verboseLevel int

func main() {
	flag.Parse()
	var strDeviceIds []string = strings.Split(strDeviceId, ",")
	var deviceNum int = len(strDeviceIds)
	var deviceIds []DeviceId
	for i := 0; i < deviceNum; i++{
		var lock sync.Mutex
		v, error := strconv.Atoi(strDeviceIds[i])
		if error != nil || v < 0{
			fmt.Println("parse deviceIds error ", error)
			return
		}
		deviceIds = append(deviceIds, DeviceId{lock, (uint32)(v), 0, 0})
	}

	if help {
		fmt.Println("Usage:\ngo run miner.go -r remote -a account -c gpu\nexample:go run miner.go -r localhost:8009 -a 0xc3d7a1ef810983847510542edfd5bc5551a6321c")
	} else {
		fmt.Println(account, remote)
	}

	var cm Miner = &Cortex{
		account:      account,
		server:       remote,
		deviceIds:     deviceIds,
		verboseLevel: uint(verboseLevel),
	}

	cm.Mining()
}
