// Copyright 2020 The CortexTheseus Authors
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
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

package compress

import (
	"bytes"
	"compress/gzip"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/golang/snappy"
	"io"
	"time"
)

func UnzipData(data []byte) (resData []byte, err error) {
	start := time.Now()
	defer func() { log.Info("Unzip data", "cost", time.Since(start)) }()
	b := bytes.NewBuffer(data)
	var r io.Reader
	r, err = gzip.NewReader(b)
	if err != nil {
		return
	}

	var resB bytes.Buffer
	_, err = resB.ReadFrom(r)
	if err != nil {
		return
	}

	resData = resB.Bytes()

	return
}

func ZipData(data []byte) (compressedData []byte, err error) {
	start := time.Now()
	defer func() { log.Info("Zip data", "cost", time.Since(start)) }()
	var b bytes.Buffer
	gz := gzip.NewWriter(&b)

	_, err = gz.Write(data)
	if err != nil {
		return
	}

	if err = gz.Flush(); err != nil {
		return
	}

	if err = gz.Close(); err != nil {
		return
	}

	compressedData = b.Bytes()

	return
}

func SnappyEncode(data []byte) ([]byte, error) {
	start := time.Now()
	defer func() { log.Info("Snappy encode", "cost", time.Since(start)) }()

	return snappy.Encode(nil, data), nil
}

func SnappyDecode(data []byte) ([]byte, error) {
	start := time.Now()
	defer func() { log.Info("Snappy decode", "cost", time.Since(start)) }()
	res, err := snappy.Decode(nil, data)
	if err != nil {
		return nil, err
	}
	return res, nil
}
