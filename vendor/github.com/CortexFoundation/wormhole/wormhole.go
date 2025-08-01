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
	"context"
	"strings"
	"sync"
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
		cl: resty.New().SetTimeout(time.Second * 5),
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

func (wh *Wormhole) healthCheckWithContext(ctx context.Context, tracker string) error {
	done := make(chan error, 1)

	go func() {
		done <- wh.healthCheck(tracker)
	}()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		return err
	}
}

func (wh *Wormhole) BestTrackers() (ret []string) {
	log.Info("Global trackers loading ... ...")

	seen := make(map[string]struct{})

	for _, ur := range BestTrackerUrl {
		log.Debug("Fetch trackers", "url", ur)

		resp, err := wh.cl.R().Get(ur)
		if err != nil || resp == nil || len(resp.String()) == 0 {
			log.Warn("Global tracker lost", "err", err)
			continue
		}

		lines := strings.Split(resp.String(), "\n")
		retCh := make(chan string, len(lines))
		start := mclock.Now()

		var wg sync.WaitGroup
		for _, line := range lines {
			tracker := strings.TrimSpace(line)
			if tracker == "" {
				continue
			}
			if _, exists := seen[tracker]; exists {
				continue
			}
			seen[tracker] = struct{}{}
			wg.Add(1)

			go func(t string) {
				defer wg.Done()

				ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
				defer cancel()

				if err := wh.healthCheckWithContext(ctx, t); err == nil {
					retCh <- t
				}
			}(tracker)
		}

		// close retCh when all health checks are done
		go func() {
			wg.Wait()
			close(retCh)
		}()

		for t := range retCh {
			log.Info("Healthy tracker", "url", t, "latency", common.PrettyDuration(time.Duration(mclock.Now())-time.Duration(start)))
			ret = append(ret, t)
			if len(ret) >= CAP {
				//return ret
			}
		}

		log.Info("Current global trackers found", "size", len(ret))
		if len(ret) >= CAP {
			break
		}
	}

	return ret
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
