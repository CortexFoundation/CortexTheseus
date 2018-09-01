package main

import (
	"../../monitor"
	download "../../manager"
	"flag"
	"fmt"
	"log"
	"time"
)

func main() {
	flag_addr := flag.String("a", "0.0.0.0:34444", "json-rpc uri")
	storageDir := flag.String("d", "/home/lizhen/storage", "storage path")
	trackerURI := flag.String("t", "http://47.52.39.170:5008/announce", "tracker uri")
	flag.Parse()
	dlCilent := download.NewTorrentManager(*storageDir)
	dlCilent.SetTrackers([]string{*trackerURI})
	m := monitor.NewHttpMonitor(*flag_addr)
	m.Initilize()
	strList := make([]string, 0)
	strMap := make(map[string]interface{})
	if false {
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
	} else {
		mURI := "magnet:?xt=urn:btih:137B5742C8B4E9CBE64109E8CF5665383ABD8DF3&dn=data&tr=http%3a%2f%2flocalhost%3a8000%2fannounce"
		dlCilent.AddMagnet(mURI)
		var tick int64 = 0;
		for {
			time.Sleep(time.Second * 1)
			tick += 1
			if tick % 10 == 0 {
				dlCilent.UpdateMagnet(mURI, tick * 1024)
			}
		}
	}
	m.Finalize()
}
