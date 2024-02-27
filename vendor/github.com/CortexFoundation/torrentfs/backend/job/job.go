// Copyright 2023 The CortexTheseus Authors
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

package job

import (
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"

	"github.com/CortexFoundation/torrentfs/backend/caffe"
	//"sync"
)

var seq atomic.Uint64

type Job struct {
	id       uint64
	status   int
	category int
	ref      *caffe.Torrent

	//wg sync.WaitGroup
	start mclock.AbsTime
}

func New(_ref *caffe.Torrent) *Job {
	job := new(Job)
	job.id = seq.Add(1)
	job.ref = _ref
	job.start = mclock.Now()
	return job
}

func (j *Job) Birth() mclock.AbsTime {
	return j.start
}

func (j *Job) ID() uint64 {
	return j.id
}

func (j *Job) Category() int {
	return j.category
}

func (j *Job) Status() int {
	return j.status
}

func (j *Job) End() {
}

func (j *Job) Ref() *caffe.Torrent {
	return j.ref
}

func SEQ() uint64 {
	return seq.Load()
}

func (j *Job) Completed(fn func(t *caffe.Torrent) bool) (result chan bool) {
	result = make(chan bool)
	go func() {
		tick := time.NewTicker(time.Second * 1)
		defer tick.Stop()
		for {
			select {
			case <-tick.C:
				if fn(j.ref) {
					/*if j.ref.IsSeeding() {
						result <- true
					} else {
						result <- false
					}*/
					result <- j.ref.BytesRequested() <= j.ref.BytesCompleted()
					return
				} else {
					log.Trace("Waiting ... ...", "ih", j.ref.InfoHash())
				}
			case <-result:
				log.Info("Job channel closed", "ih", j.ref.InfoHash(), "id", j.id, "status", j.ref.Status(), "complete", j.ref.BytesCompleted(), "miss", j.ref.BytesMissing(), "request", j.ref.BytesRequested())
				return
			}
		}
	}()

	return
}
