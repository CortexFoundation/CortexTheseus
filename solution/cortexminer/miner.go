package cortexminer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"os"
	"time"

	"github.com/CortexFoundation/CortexTheseus/solution/config"

	"plugin"
	"strconv"
)

func checkError(err error, func_name string) {
	if err != nil {
		log.Println(func_name, err.Error())
	}
}

func (cm *Cortex) read(msgCh chan map[string]interface{}) {
	if cm.reader == nil {
		return
	}

	tmp, isPrefix, err := cm.reader.ReadLine()
	if err == io.EOF {
		log.Println("Tcp disconnect")
		cm.consta.state = false
		for {
			if cm.consta.state {
				log.Println("Tcp reconnect successfully")
				return
			}
			log.Println("Tcp reconnecting")
			time.Sleep(1 * time.Second)
		}
	}

	if err != nil {
		return
	}

	if isPrefix == false {
	}
	var repObj map[string]interface{}
	err = json.Unmarshal(tmp, &repObj)
	if err != nil {
		return
	}
	msgCh <- repObj
}

func (cm *Cortex) write(reqObj ReqObj) {
	req, err := json.Marshal(reqObj)
	if err != nil {
		return
	}

	req = append(req, uint8('\n'))
	if cm.conn != nil {
		go cm.conn.Write(req)
	}
}

func (cm *Cortex) init(tcpCh chan bool) {
	log.Println("Cortex connecting")
	tcpAddr, err := net.ResolveTCPAddr("tcp", cm.param.Server)
	if err != nil {
		tcpCh <- false
		log.Println("Cortex connecting", err)
		return
	}

	cm.conn, err = net.DialTCP("tcp", nil, tcpAddr)
	if err != nil {
		tcpCh <- false
		log.Println("Cortex dial", err)
		return
	}

	if cm.conn == nil {
		tcpCh <- false
		log.Println("Cortex connect is null")
		return
	}
	cm.conn.SetKeepAlive(true)
	cm.conn.SetNoDelay(true)
	cm.reader = bufio.NewReader(cm.conn)
	tcpCh <- true
	log.Println("Cortex connect successfully")
}

//	miner login to mining pool
func (cm *Cortex) login(loginCh chan bool) {
	log.Println("Cortex login ...")
	var reqLogin = ReqObj{
		Id:      73,
		Jsonrpc: "2.0",
		Method:  "ctxc_submitLogin",
		Params:  []string{cm.param.Account},
	}
	cm.write(reqLogin)
	cm.getWork()
	log.Println("Cortex login successfully")
	loginCh <- true
}

//	get mining task
func (cm *Cortex) getWork() {
	req := ReqObj{
		Id:      100,
		Jsonrpc: "2.0",
		Method:  "ctxc_getWork",
		Params:  []string{""},
	}
	cm.write(req)
}

//	submit task
func (cm *Cortex) submit(sol config.Task) {
	var reqSubmit = ReqObj{
		Id:      73,
		Jsonrpc: "2.0",
		Method:  "ctxc_submitWork",
		Params:  []string{sol.Nonce, sol.Header, sol.Solution},
	}
	cm.write(reqSubmit)
}

var minerPlugin *plugin.Plugin

const PLUGIN_PATH string = "plugins/"
const PLUGIN_POST_FIX string = "_helper.so"

//	cortex mining
func (cm *Cortex) Mining() {
	var iDeviceIds []uint32
	for i := 0; i < len(cm.deviceInfos); i++ {
		iDeviceIds = append(iDeviceIds, cm.deviceInfos[i].DeviceId)
	}

	var minerName string = ""
	if cm.param.Cpu == true {
		minerName = "cpu"
	} else if cm.param.Cuda == true {
		minerName = "cuda"
	} else if cm.param.Opencl == true {
		minerName = "opencl"
	} else {
		os.Exit(0)
	}

	var err error
	minerPlugin, err = plugin.Open(PLUGIN_PATH + minerName + PLUGIN_POST_FIX)
	m, err := minerPlugin.Lookup("CuckooInitialize")
	if err != nil {
		panic(err)
	}
	m.(func([]uint32, uint32, config.Param))(iDeviceIds, (uint32)(len(iDeviceIds)), *cm.param)
	go func() {
		for {
			cm.printHashRate()
			time.Sleep(3 * time.Second)
		}
	}()

	tcpCh := make(chan bool)
	loginCh := make(chan bool)
	startCh := make(chan bool)
	init := true
	go func(start chan bool) {
		for {
			if !cm.consta.state {
				go cm.init(tcpCh)
				select {
				case suc := <-tcpCh:
					if !suc {
						continue
					}
				}

				go cm.login(loginCh)
				select {
				case suc := <-loginCh:
					if !suc {
						continue
					}
					cm.consta.state = true
					if init {
						init = false
						start <- true
					}
				}
			}
			time.Sleep(100 * time.Millisecond)
		}
	}(startCh)

	select {
	case suc := <-startCh:
		if suc {
			log.Println("Start mining")
		}
	}

	miningCh := make(chan string)
	go cm.mining(miningCh)
	select {
	case quit := <-miningCh:
		if quit == "quit" {
		}
	}
}

func (cm *Cortex) printHashRate() {
	var devCount = len(cm.deviceInfos)
	var fanSpeeds []uint32
	var temperatures []uint32
	m, err := minerPlugin.Lookup("Monitor")
	if err != nil {
		panic(err)
	}
	fanSpeeds, temperatures = m.(func(uint32) ([]uint32, []uint32))(uint32(devCount))
	var total_solutions int64 = 0
	for dev := 0; dev < devCount; dev++ {
		var dev_id = cm.deviceInfos[dev].DeviceId
		gps := (float32(1000.0*cm.deviceInfos[dev].Gps) / float32(cm.deviceInfos[dev].Use_time))
		if cm.deviceInfos[dev].Use_time > 0 && cm.deviceInfos[dev].Solution_count > 0 {
			cm.deviceInfos[dev].Hash_rate = (float32(1000.0*cm.deviceInfos[dev].Solution_count) / float32(cm.deviceInfos[dev].Use_time))
			log.Println(fmt.Sprintf("\033[0;%dmGPU%d GPS=%.4f, hash rate=%.4f, find solutions:%d, fan=%d%%, t=%dC\033[0m", 32+(dev%2*2), dev_id, gps, cm.deviceInfos[dev].Hash_rate, cm.deviceInfos[dev].Solution_count, fanSpeeds[dev], temperatures[dev]))
			total_solutions += cm.deviceInfos[dev].Solution_count
		} else {
			log.Println(fmt.Sprintf("\033[0;%dmGPU%d GPS=%.4f, hash rate=Inf, find solutions: 0, fan=%d%%, t=%dC\033[0m", 32+(dev%2*2), dev_id, gps, fanSpeeds[dev], temperatures[dev]))
		}
	}
	log.Println(fmt.Sprintf("\033[0;36mfind total solutions : %d, share accpeted : %d, share rejected : %d\033[0m", total_solutions, cm.share_accepted, cm.share_rejected))
}

func readNonce() (ret []uint64) {
	fi, err := os.Open("nonces.txt")
	if err != nil {
		log.Println("Error:", err)
	}
	defer fi.Close()

	br := bufio.NewReader(fi)
	for {
		a, _, c := br.ReadLine()
		if c == io.EOF {
			break
		}
		var strNonce string = string(a)
		nonce, _ := strconv.ParseInt(strNonce, 10, 64)
		ret = append(ret, uint64(nonce))
	}
	return ret
}

func (cm *Cortex) mining(quitCh chan string) {
	var taskHeader, taskNonce, taskDifficulty string
	var THREAD int = (int)(len(cm.deviceInfos))
	rand.Seed(time.Now().UTC().UnixNano())
	solChan := make(chan config.Task, THREAD)
	taskChan := make(chan config.Task, THREAD)

	m, err := minerPlugin.Lookup("RunSolver")
	if err != nil {
		panic(err)
	}
	m.(func(int, []config.DeviceInfo, config.Param, chan config.Task, chan config.Task, bool) (uint32, [][]uint32))(THREAD, cm.deviceInfos, *cm.param, taskChan, solChan, cm.consta.state)
	go func() {
		for {
			select {
			case sol := <-solChan:
				if sol.Header == config.CurrentTask.TaskQ.Header {
					cm.submit(sol)
				}
			}
		}
	}()

	msgCh := make(chan map[string]interface{}, THREAD)
	go func() {
		for {
			cm.read(msgCh)
		}
	}()

	//go cm.getWork()

	for {
		select {
		case msg := <-msgCh:
			if cm.consta.state == false || msg == nil {
				continue
			}
			reqId, _ := msg["id"].(float64)
			result, _ := msg["result"].(bool)
			if uint32(reqId) == 73 {
				if result {
					cm.share_accepted += 1
				} else {
					cm.share_rejected += 1
				}
			}
			if uint32(reqId) == 100 || uint32(reqId) == 0 {
				workInfo, _ := msg["result"].([]interface{})
				if len(workInfo) >= 3 {
					taskHeader, taskNonce, taskDifficulty = workInfo[0].(string), workInfo[1].(string), workInfo[2].(string)
					log.Println("Get Work in task: ", taskHeader, taskDifficulty)
					config.CurrentTask.Lock.Lock()
					config.CurrentTask.TaskQ.Nonce = taskNonce
					config.CurrentTask.TaskQ.Header = taskHeader
					config.CurrentTask.TaskQ.Difficulty = taskDifficulty
					config.CurrentTask.Lock.Unlock()
					taskChan <- config.CurrentTask.TaskQ
				}
			}
		}
	}
	//quitCh <- "quit"
}
