package main

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/consensus/cuckoo"
	"github.com/ethereum/go-ethereum/core/types"
	_ "log"
	"math/rand"
	"net"
	"os"
	"time"

	// "strconv"
	"sync"
)

type Task struct {
	Header     string
	Nonce      string
	Solution   string
	Difficulty string
}

type Work struct {
	Header     string
	Difficulty string
	nonce      uint64
	step       uint64
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
	var testFixTask bool = false
	if testFixTask {
		fmt.Println("testFixTask = ", testFixTask)

	}

	type TaskWrapper struct {
		Lock  sync.Mutex
		TaskQ Task
	}
	var currentTask TaskWrapper

	var THREAD uint = 5
	cuckoo.CuckooInit(8)
	var taskHeader, taskNonce, taskDifficulty string
	//-------- connect to server -------------
	// var server = "139.196.32.192:8009"
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
	_ = read(reader)

	solChanG := make(chan Task, THREAD)
	workChanG := make(chan Work, THREAD)
	for nthread := 0; nthread < int(THREAD); nthread++ {
		go func(tidx uint32, solChan chan Task, workChan chan Work) {
			for {
				currentTask.Lock.Lock()
				task := currentTask.TaskQ
				fmt.Println("get work--: ", task.Header, task.Nonce, task.Difficulty)
				currentTask.Lock.Unlock()
				taskDifficulty = task.Difficulty
				if len(task.Difficulty) == 0 {
					time.Sleep(100 * time.Millisecond)
					continue
				}
				tgtDiff := common.HexToHash(taskDifficulty[2:])
				header, _ := hex.DecodeString(taskHeader[2:])
				var result types.BlockSolution
				curNonce := uint32(rand.Int31())
				// r := cuckoo.CuckooSolve(&tmpHeader[0], 32, uint32(start), &result[0], &result_len, &targetMinerTest[0], &result_hash[0])
				solutionsHolder := make([]uint32, 128)
				if true {
					fmt.Println("task: ", header[:], curNonce)
					status, solLength, numSol := cuckoo.CuckooFindSolutions(header[:], curNonce, &solutionsHolder)
					if status != 0 {
						// fmt.Println("result: ", status, solLength, numSol, solutionsHolder)
						for solIdx := uint32(0); solIdx < numSol; solIdx++ {
							var sol types.BlockSolution
							copy(sol[:], solutionsHolder[solIdx*solLength:(solIdx+1)*solLength])
							sha3hash := common.BytesToHash(cuckoo.Sha3Solution(&sol))
							fmt.Println(curNonce, "\n sol hash: ", hex.EncodeToString(sha3hash.Bytes()), "\n tgt hash: ", hex.EncodeToString(tgtDiff.Bytes()))
							if sha3hash.Big().Cmp(tgtDiff.Big()) <= 0 {
								result = sol
								nonceStr := common.Uint64ToHexString(uint64(curNonce))
								digest := common.Uint32ArrayToHexString([]uint32(result[:]))
								ok, _ := cuckoo.CuckooVerifyHeaderNonceSolutionsDifficulty(header[:], curNonce, &sol)
								if !ok {
									fmt.Println("verify failed", header[:], curNonce, &sol)
								}
								solChan <- Task{Nonce: nonceStr, Header: taskHeader, Solution: digest}
								break
							}
						}
					}
				} else {
					var sol types.BlockSolution
					sol[0] = curNonce
					sha3hash := common.BytesToHash(cuckoo.Sha3Solution(&sol))
					// fmt.Println(curNonce, "\n sol hash: ", hex.EncodeToString(sha3hash.Bytes()), "\n tgt hash: ", hex.EncodeToString(tgtDiff.Bytes()))
					if sha3hash.Big().Cmp(tgtDiff.Big()) <= 0 {
						result = sol
						nonceStr := common.Uint64ToHexString(uint64(curNonce))
						digest := common.Uint32ArrayToHexString([]uint32(result[:]))
						solChan <- Task{Nonce: nonceStr, Header: taskHeader, Solution: digest}
						break
					}
				}
			}
		}(uint32(nthread), solChanG, workChanG)
	}

	time.Sleep(2 * time.Second)
	for {
		select {
		case sol := <-solChanG:
			fmt.Println("Found: ", sol)
			var reqSubmit = ReqObj{
				Id:      73,
				Jsonrpc: "2.0",
				Method:  "eth_submitWork",
				Params:  []string{sol.Nonce, sol.Header, sol.Header, sol.Solution},
			}
			write(reqSubmit, conn)
			_ = read(reader)
		case <-time.After(2000 * time.Millisecond):
			var reqGetwork = ReqObj{
				Id:      73,
				Jsonrpc: "2.0",
				Method:  "eth_getWork",
				Params:  []string{""},
			}
			write(reqGetwork, conn)
			work := read(reader)
			workInfo, _ := work["result"].([]interface{})
			if len(workInfo) >= 3 {
				taskHeader, taskNonce, taskDifficulty = workInfo[0].(string), workInfo[1].(string), workInfo[2].(string)
				fmt.Println("get work: ", taskHeader, taskNonce, taskDifficulty)
				currentTask.Lock.Lock()
				currentTask.TaskQ.Nonce = taskNonce
				currentTask.TaskQ.Header = taskHeader
				currentTask.TaskQ.Difficulty = taskDifficulty
				currentTask.Lock.Unlock()
			}
		}
	}
	cuckoo.CuckooFinalize()
}
