package main

import (
	"flag"
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/solution/config"
	"github.com/CortexFoundation/CortexTheseus/solution/cortexminer"
	"log"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

func init() {
	flag.BoolVar(&help, "help", false, "show help")
	flag.StringVar(&remote, "pool_uri", "47.91.2.19:8009", "mining pool address")
	flag.StringVar(&account, "account", "0xc3d7a1ef810983847510542edfd5bc5551a6321c", "miner accounts")
	flag.StringVar(&strDeviceId, "devices", "0", "which GPU device use for mining")
	flag.IntVar(&verboseLevel, "verbosity", 0, "verbosity level")
	flag.StringVar(&algorithm, "algorithm", "cuckoo", "use cuckoo or cuckaroo")
	flag.IntVar(&threads, "threads", 1, "how many cpu threads")
	flag.BoolVar(&cpu, "cpu", false, "use cpu miner")
	flag.BoolVar(&cuda, "cuda", false, "use cuda miner")
	flag.BoolVar(&opencl, "opencl", false, "use opencl miner")

	/*
		cfg, err := goconfig.LoadConfigFile("miner.ini")
		if err == nil {
			remote, err = cfg.GetValue("server", "addr")
			checkError(err, "init()")
			account, err = cfg.GetValue("mining", "account")
			checkError(err, "init()")

			useGPU, err = cfg.Bool("mining", "useGPU")
			checkError(err, "init()")
			strDeviceId, err = cfg.GetValue("mining", "devices")
			checkError(err, "init()")
			verboseLevel, err = cfg.Int("mining", "verboselevel")
			checkError(err, "init()")
			algorithm, err = cfg.GetValue("mining", "algorithm")
			checkError(err, "init()")
		}
	*/

	fmt.Printf("**************************************************************\n")
	fmt.Printf("**\t\tCortex GPU Miner\t\t\t**\n")
	fmt.Printf("**************************************************************\n")
}

var help bool
var remote string = ""
var account string = ""
var strDeviceId string = ""
var verboseLevel int = 0
var algorithm string = ""
var miner_algorithm int
var threads int
var cpu bool
var cuda bool
var opencl bool

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU() * 4)
	flag.Parse()
	if algorithm == "cuckoo" {
		miner_algorithm = 0
	} else if algorithm == "cuckaroo" {
		miner_algorithm = 1
	} else {
		log.Fatalf("no support algorithm: "+ algorithm)
		os.Exit(1)
	}

	var strDeviceIds []string = strings.Split(strDeviceId, ",")
	var deviceNum int = len(strDeviceIds)
	var deviceInfos []config.DeviceInfo
	var start_time int64 = time.Now().UnixNano() / 1e6
	for i := 0; i < deviceNum; i++ {
		var lock sync.Mutex
		v, error := strconv.Atoi(strDeviceIds[i])
		if error != nil || v < 0 {
			fmt.Println("parse deviceIds error ", error)
			return
		}
		var deviceInfo config.DeviceInfo
		deviceInfos = append(deviceInfos, deviceInfo.New(lock, (uint32)(v), start_time, 0, 0, 0, 0))
	}
	if help {
		fmt.Println("Usage:\ngo run miner.go -r remote -a account -c gpu\nexample:go run miner.go -r localhost:8009 -a 0xc3d7a1ef810983847510542edfd5bc5551a6321c")
	} else {
		fmt.Println(account, remote)
	}

	var param config.Param
	var cortex cortexminer.Cortex
	cm := cortex.New(
		deviceInfos,
		param.New(remote, account, uint(verboseLevel), miner_algorithm, threads, cpu, cuda, opencl))

	cm.Mining()
}
