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
	defer log.Info("Unzip data", "cost", time.Since(start))
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
	defer log.Info("Zip data", "cost", time.Since(start))
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
	defer log.Info("Snappy encode", "cost", time.Since(start))

	return snappy.Encode(nil, data), nil
}

func SnappyDecode(data []byte) ([]byte, error) {
	start := time.Now()
	defer log.Info("Snappy decode", "cost", time.Since(start))
	res, err := snappy.Decode(nil, data)
	if err != nil {
		return nil, err
	}
	return res, nil
}
