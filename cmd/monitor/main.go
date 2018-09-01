package main

import "../../monitor"
import (
	"flag"
	"fmt"
	"log"
	"time"
)

func main() {
	flag_addr := flag.String("a", "0.0.0.0:34444", "json-rpc uri")
	m := monitor.NewHttpMonitor(*flag_addr)
	m.Initilize()
	strList := make([]string, 0)
	strMap := make(map[string]interface{})
	for {
		time.Sleep(time.Second * 1)
		log.Println(strList)
		now := time.Now().UnixNano()
		strList = append(strList, fmt.Sprintf("%d", time.Now()))
		if now%2 == 0 {
			strMap[fmt.Sprintf("%d", now)] = now
		} else {
			strMap[fmt.Sprintf("%d", now)] = []string{fmt.Sprintf("%d", now)}
		}

		m.Update("strList", strList)
		m.Update("time", time.Now())
		m.Update("a.b.c", strMap)
	}
	m.Finalize()
}
