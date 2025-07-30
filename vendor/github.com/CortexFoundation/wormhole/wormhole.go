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
	"slices"
	"strings"
	//"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	mapset "github.com/deckarep/golang-set/v2"
	resty "github.com/go-resty/resty/v2"
)

type Wormhole struct {
	cl *resty.Client
}

func New() *Wormhole {
	return &Wormhole{
		cl: resty.New().SetTimeout(time.Second * 10),
	}
}

func (wh *Wormhole) Tunnel(hash string) error {
	log.Debug("Wormhole tunnel", "hash", hash)
	for _, worm := range Wormholes {
		if _, err := wh.cl.R().Post(worm + hash); err != nil {
			log.Error("Wormhole err", "err", err, "worm", worm, "hash", hash)
		}
	}

	return nil
}

func (wh *Wormhole) BestTrackers() (ret []string) {
	defer wh.cl.SetTimeout(time.Second * 10)

	log.Info("Global trackers loading ... ...")

	for _, ur := range BestTrackerUrl {
		log.Debug("Fetch trackers", "url", ur)
		resp, err := wh.cl.R().Get(ur)

		if err != nil || resp == nil || len(resp.String()) == 0 {
			log.Warn("Global tracker lost", "err", err)
			continue
		}

		wh.cl.SetTimeout(time.Millisecond * 2000)

		var (
			str      = strings.Split(resp.String(), "\n\n")
			retCh    = make(chan string, len(str))
			failedCh = make(chan string, len(str))
			start    = mclock.Now()
			count    = 0
		)
		for _, s := range str {
			if slices.Contains(ret, s) {
				continue
			}
			count++
			go func(ss string) {
				if err := wh.healthCheck(ss); err == nil {
					retCh <- ss
				} else {
					failedCh <- ss
				}
			}(s)
		}

		for i := 0; i < count; i++ {
			select {
			case x := <-retCh:
				log.Debug("Healthy tracker", "url", x, "latency", common.PrettyDuration(time.Duration(mclock.Now())-time.Duration(start)))
				ret = append(ret, x)
			case x := <-failedCh:
				// TODO
				log.Debug("Unhealthy tracker", "url", x, "latency", common.PrettyDuration(time.Duration(mclock.Now())-time.Duration(start)))

			}
		}

		log.Info("Current global trackers found", "size", len(ret))

		if len(ret) > CAP {
			return
		}
		wh.cl.SetTimeout(time.Second * 10)
	}

	return
}

func (wh *Wormhole) ColaList() mapset.Set[string] {
	m := mapset.NewSet[string]()
	for _, url := range ColaUrl {
		resp, err := wh.cl.R().Get(url)

		if err != nil {
			log.Warn("Cola lost", "err", err)
			continue
		}

		str := strings.Split(resp.String(), "\n\n")
		for _, s := range str {
			log.Info("Cola", "ih", s)
			m.Add(s)
		}
	}

	return m
}
