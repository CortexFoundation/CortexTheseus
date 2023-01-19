// Copyright 2020-2021 InfluxData, Inc. All rights reserved.
// Use of this source code is governed by MIT
// license that can be found in the LICENSE file.

package api

import (
	"context"
	"strings"
	"time"

	http2 "github.com/influxdata/influxdb-client-go/v2/api/http"
	"github.com/influxdata/influxdb-client-go/v2/api/write"
	"github.com/influxdata/influxdb-client-go/v2/internal/log"
	iwrite "github.com/influxdata/influxdb-client-go/v2/internal/write"
)

// WriteAPI is Write client interface with non-blocking methods for writing time series data asynchronously in batches into an InfluxDB server.
// WriteAPI can be used concurrently.
// When using multiple goroutines for writing, use a single WriteAPI instance in all goroutines.
type WriteAPI interface {
	// WriteRecord writes asynchronously line protocol record into bucket.
	// WriteRecord adds record into the buffer which is sent on the background when it reaches the batch size.
	// Blocking alternative is available in the WriteAPIBlocking interface
	WriteRecord(line string)
	// WritePoint writes asynchronously Point into bucket.
	// WritePoint adds Point into the buffer which is sent on the background when it reaches the batch size.
	// Blocking alternative is available in the WriteAPIBlocking interface
	WritePoint(point *write.Point)
	// Flush forces all pending writes from the buffer to be sent
	Flush()
	// Errors returns a channel for reading errors which occurs during async writes.
	// Must be called before performing any writes for errors to be collected.
	// The chan is unbuffered and must be drained or the writer will block.
	Errors() <-chan error
}

// WriteAPIImpl provides main implementation for WriteAPI
type WriteAPIImpl struct {
	service     *iwrite.Service
	writeBuffer []string

	errCh        chan error
	writeCh      chan *iwrite.Batch
	bufferCh     chan string
	writeStop    chan struct{}
	bufferStop   chan struct{}
	bufferFlush  chan struct{}
	doneCh       chan struct{}
	bufferInfoCh chan writeBuffInfoReq
	writeInfoCh  chan writeBuffInfoReq
	writeOptions *write.Options
}

type writeBuffInfoReq struct {
	writeBuffLen int
}

// NewWriteAPI returns new non-blocking write client for writing data to  bucket belonging to org
func NewWriteAPI(org string, bucket string, service http2.Service, writeOptions *write.Options) *WriteAPIImpl {
	w := &WriteAPIImpl{
		service:      iwrite.NewService(org, bucket, service, writeOptions),
		writeBuffer:  make([]string, 0, writeOptions.BatchSize()+1),
		writeCh:      make(chan *iwrite.Batch),
		bufferCh:     make(chan string),
		bufferStop:   make(chan struct{}),
		writeStop:    make(chan struct{}),
		bufferFlush:  make(chan struct{}),
		doneCh:       make(chan struct{}),
		bufferInfoCh: make(chan writeBuffInfoReq),
		writeInfoCh:  make(chan writeBuffInfoReq),
		writeOptions: writeOptions,
	}

	go w.bufferProc()
	go w.writeProc()

	return w
}

// Errors returns a channel for reading errors which occurs during async writes.
// Must be called before performing any writes for errors to be collected.
// The chan is unbuffered and must be drained or the writer will block.
func (w *WriteAPIImpl) Errors() <-chan error {
	if w.errCh == nil {
		w.errCh = make(chan error)
	}
	return w.errCh
}

// Flush forces all pending writes from the buffer to be sent
func (w *WriteAPIImpl) Flush() {
	w.bufferFlush <- struct{}{}
	w.waitForFlushing()
}

func (w *WriteAPIImpl) waitForFlushing() {
	for {
		w.bufferInfoCh <- writeBuffInfoReq{}
		writeBuffInfo := <-w.bufferInfoCh
		if writeBuffInfo.writeBuffLen == 0 {
			break
		}
		log.Info("Waiting buffer is flushed")
		<-time.After(time.Millisecond)
	}
	for {
		w.writeInfoCh <- writeBuffInfoReq{}
		writeBuffInfo := <-w.writeInfoCh
		if writeBuffInfo.writeBuffLen == 0 {
			break
		}
		log.Info("Waiting buffer is flushed")
		<-time.After(time.Millisecond)
	}
}

func (w *WriteAPIImpl) bufferProc() {
	log.Info("Buffer proc started")
	ticker := time.NewTicker(time.Duration(w.writeOptions.FlushInterval()) * time.Millisecond)
x:
	for {
		select {
		case line := <-w.bufferCh:
			w.writeBuffer = append(w.writeBuffer, line)
			if len(w.writeBuffer) == int(w.writeOptions.BatchSize()) {
				w.flushBuffer()
			}
		case <-ticker.C:
			w.flushBuffer()
		case <-w.bufferFlush:
			w.flushBuffer()
		case <-w.bufferStop:
			ticker.Stop()
			w.flushBuffer()
			break x
		case buffInfo := <-w.bufferInfoCh:
			buffInfo.writeBuffLen = len(w.bufferInfoCh)
			w.bufferInfoCh <- buffInfo
		}
	}
	log.Info("Buffer proc finished")
	w.doneCh <- struct{}{}
}

func (w *WriteAPIImpl) flushBuffer() {
	if len(w.writeBuffer) > 0 {
		log.Info("sending batch")
		batch := iwrite.NewBatch(buffer(w.writeBuffer), w.writeOptions.RetryInterval())
		w.writeCh <- batch
		w.writeBuffer = w.writeBuffer[:0]
	}
}

func (w *WriteAPIImpl) writeProc() {
	log.Info("Write proc started")
x:
	for {
		select {
		case batch := <-w.writeCh:
			err := w.service.HandleWrite(context.Background(), batch)
			if err != nil && w.errCh != nil {
				w.errCh <- err
			}
		case <-w.writeStop:
			log.Info("Write proc: received stop")
			break x
		case buffInfo := <-w.writeInfoCh:
			buffInfo.writeBuffLen = len(w.writeCh)
			w.writeInfoCh <- buffInfo
		}
	}
	log.Info("Write proc finished")
	w.doneCh <- struct{}{}
}

// Close finishes outstanding write operations,
// stop background routines and closes all channels
func (w *WriteAPIImpl) Close() {
	if w.writeCh != nil {
		// Flush outstanding metrics
		w.Flush()

		// stop and wait for buffer proc
		close(w.bufferStop)
		<-w.doneCh

		close(w.bufferFlush)
		close(w.bufferCh)

		// stop and wait for write proc
		close(w.writeStop)
		<-w.doneCh

		close(w.writeCh)
		close(w.writeInfoCh)
		close(w.bufferInfoCh)
		w.writeCh = nil

		// close errors if open
		if w.errCh != nil {
			close(w.errCh)
			w.errCh = nil
		}
	}
}

// WriteRecord writes asynchronously line protocol record into bucket.
// WriteRecord adds record into the buffer which is sent on the background when it reaches the batch size.
// Blocking alternative is available in the WriteAPIBlocking interface
func (w *WriteAPIImpl) WriteRecord(line string) {
	b := []byte(line)
	b = append(b, 0xa)
	w.bufferCh <- string(b)
}

// WritePoint writes asynchronously Point into bucket.
// WritePoint adds Point into the buffer which is sent on the background when it reaches the batch size.
// Blocking alternative is available in the WriteAPIBlocking interface
func (w *WriteAPIImpl) WritePoint(point *write.Point) {
	line, err := w.service.EncodePoints(point)
	if err != nil {
		log.Errorf("point encoding error: %s\n", err.Error())
		if w.errCh != nil {
			w.errCh <- err
		}
	} else {
		w.bufferCh <- line
	}
}

func buffer(lines []string) string {
	return strings.Join(lines, "")
}
