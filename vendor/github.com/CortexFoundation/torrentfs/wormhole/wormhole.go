// Copyright 2022 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>

package wormhole

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs/params"
	"net"
	"net/url"

	resty "github.com/go-resty/resty/v2"

	mapset "github.com/deckarep/golang-set/v2"

	"strings"
	"time"
)

var (
	client *resty.Client = resty.New().SetTimeout(time.Second * 10)
)

func Tunnel(hash string) error {
	log.Debug("Wormhole tunnel", "hash", hash)
	for _, worm := range Wormholes {
		if _, err := client.R().Post(worm + hash); err != nil {
			log.Error("Wormhole err", "err", err, "worm", worm, "hash", hash)
		}
	}

	return nil
}

func BestTrackers() (ret []string) {
	for _, ur := range params.BestTrackerUrl {
		resp, err := client.R().Get(ur)

		if err != nil {
			log.Warn("Global tracker lost", "err", err)
			continue
		}

		str := strings.Split(resp.String(), "\n\n")
		for _, s := range str {
			if len(ret) < CAP { //&& strings.HasPrefix(s, "udp") {
				log.Debug("Global best trackers", "url", s)
				if strings.HasPrefix(s, "http") || strings.HasPrefix(s, "https") {
					response, err := client.R().Post(s)
					if err != nil || response == nil {
						log.Warn("tracker failed", "err", err)
					} else {
						ret = append(ret, s)
					}
				} else if strings.HasPrefix(s, "udp") {
					u, err := url.Parse(s)
					if err != nil {
						continue
					}
					if host, port, err := net.SplitHostPort(u.Host); err == nil {
						if err := ping(host, port); err == nil {
							ret = append(ret, s)
						} else {
							log.Warn("UDP ping err", "s", s, "err", err)
						}
					}
				} else {
					log.Warn("Other protocols trackers", "s", s)
				}
			}
		}

		if len(ret) > 0 {
			return
		}
	}

	return
}

func ColaList() mapset.Set[string] {
	m := mapset.NewSet[string]()
	for _, url := range params.ColaUrl {
		resp, err := client.R().Get(url)

		if err != nil {
			log.Warn("Cola lost", "err", err)
			continue
		}

		str := strings.Split(resp.String(), "\n\n")
		for _, s := range str {
			log.Info("Cola", "ih", s)
			//ret = append(ret, s)
			m.Add(s)
		}
	}

	return m
}

func ping(host string, port string) error {
	address := net.JoinHostPort(host, port)
	conn, err := net.DialTimeout("udp", address, 1*time.Second)
	if conn != nil {
		defer conn.Close()
	}
	return err
}
