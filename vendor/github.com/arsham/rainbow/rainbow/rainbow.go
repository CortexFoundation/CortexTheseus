// Copyright 2020 Arsham Shirvani <arshamshirvani@gmail.com>. All rights reserved.
// Use of this source code is governed by the Apache 2.0 license
// License that can be found in the LICENSE file.

// Package rainbow prints texts in beautiful rainbows in terminal. Usage is very
// simple:
//
//   import "github.com/arsham/rainbow/rainbow"
//   // ...
//   l := rainbow.Light{
//       Reader: someReader, // to read from
//       Writer: os.Stdout, // to write to
//   }
//   l.Paint() // will rainbow everything it reads from reader to writer.
//
// If you want the rainbow to be random, you can seed it this way:
//   l := rainbow.Light{
//       Reader: buf,
//       Writer: os.Stdout,
//       Seed:   rand.Int63n(256),
//   }
//
// You can also use the Light as a Writer:
//   l := rainbow.Light{
//       Writer: os.Stdout, // to write to
//       Seed:   rand.Int63n(256),
//   }
//   io.Copy(l, someReader)
package rainbow

import (
	"bytes"
	"errors"
	"io"
	"math"
	"math/rand"
	"regexp"
	"strconv"
)

var (
	// We remove all previous paintings to create a new rainbow.
	colorMatch = regexp.MustCompile("^\033" + `\[\d+(;\d+)?(;\d+)?[mK]`)

	// ErrNilWriter is returned when Light.Writer is nil.
	ErrNilWriter = errors.New("nil writer")
)

const (
	freq   = 0.1
	spread = 3
)

// Light reads data from the Writer and pains the contents to the Reader. You
// should seed it everytime otherwise you get the same results.
type Light struct {
	Reader io.Reader
	Writer io.Writer
	Seed   int64
}

// Paint returns an error if it could not copy the data.
func (l *Light) Paint() error {
	if l.Seed == 0 {
		l.Seed = rand.Int63n(256)
	}
	_, err := io.Copy(l, l.Reader)
	return err
}

// Write paints the data and writes it into l.Writer.
func (l *Light) Write(data []byte) (int, error) {
	if l.Writer == nil {
		return 0, ErrNilWriter
	}
	var (
		offset  float64
		dataLen = len(data)
		// 16 times seems to be the sweet spot.
		buf  = bytes.NewBuffer(make([]byte, 0, dataLen*16))
		seed = l.Seed
	)

	data = colorMatch.ReplaceAll(data, []byte(""))
	for _, c := range string(data) {
		switch c {
		case '\n':
			offset = 0
			seed++
			buf.WriteByte('\n')
		case '\t':
			offset++
			buf.WriteByte('\t')
		default:
			r, g, b := plotPos(float64(seed) + (offset / spread))
			colouriseWriter(buf, c, r, g, b)
			offset++
		}
	}
	_, err := l.Writer.Write(buf.Bytes())
	return dataLen, err
}

func plotPos(x float64) (red, green, blue float64) {
	red = math.Sin(freq*x)*127 + 128
	green = math.Sin(freq*x+2*math.Pi/3)*127 + 128
	blue = math.Sin(freq*x+4*math.Pi/3)*127 + 128
	return red, green, blue
}

const max = 16 + (6 * (127 + 128) / 256 * 36) + (6 * (127 + 128) / 256 * 6) + (6 * (127 + 128) / 256)

// nums is used to cache the values of strconv.Itoa(n) for better performance
// gains.
var nums = make([]string, 0, max)

func init() {
	for i := int64(0); i < max; i++ {
		nums = append(nums, strconv.FormatInt(i, 10))
	}
}

func colouriseWriter(s *bytes.Buffer, c rune, r, g, b float64) {
	s.WriteString("\033[38;5;")
	s.WriteString(nums[colour(r, g, b)])
	s.WriteByte('m')
	s.WriteRune(c)
	s.WriteString("\033[0m")
}

func colour(red, green, blue float64) uint8 {
	return 16 + baseColor(red, 36) + baseColor(green, 6) + baseColor(blue, 1)
}

func baseColor(value float64, factor uint8) uint8 {
	return uint8(6*value/256) * factor
}
