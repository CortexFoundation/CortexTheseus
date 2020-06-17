// Copyright 2016 Arsham Shirvani <arshamshirvani@gmail.com>. All rights reserved.
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
//       Seed:   int(rand.Int31n(256)),
//   }
//
// You can also use the Light as a Writer:
//   l := rainbow.Light{
//       Writer: os.Stdout, // to write to
//       Seed:   int(rand.Int31n(256)),
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
	"sync/atomic"

	"github.com/arsham/strings"
)

var (
	colorMatch = regexp.MustCompile("^\033" + `\[(\d+)(;\d+)?(;\d+)?[m|K]`)
	tabs       = []byte("\t")
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
	var skip, offset int
	if l.Writer == nil {
		return 0, ErrNilWriter
	}
	buf := &bytes.Buffer{}
	for i, c := range string(data) {
		if skip > 0 {
			skip--
			continue
		}
		switch c {
		case '\n':
			offset = 0
			atomic.AddInt64(&l.Seed, 1)
			buf.WriteByte('\n')
		case '\t':
			offset += len(tabs)
			buf.Write(tabs)
		default:
			pos := colorMatch.FindIndex(data[i:])
			if pos != nil {
				skip = pos[1] - 1
				continue
			}
			r, g, b := plotPos(float64(atomic.LoadInt64(&l.Seed)) + (float64(offset) / spread))
			w := colourise(c, r, g, b)
			buf.Write(w.Bytes())
			offset++
		}
		skip = 0
	}
	_, err := l.Writer.Write(buf.Bytes())
	return len(data), err
}

func plotPos(x float64) (int, int, int) {
	red := math.Sin(freq*x)*127 + 128
	green := math.Sin(freq*x+2*math.Pi/3)*127 + 128
	blue := math.Sin(freq*x+4*math.Pi/3)*127 + 128
	return int(red), int(green), int(blue)
}

func colourise(c rune, r, g, b int) *strings.Builder {
	s := &strings.Builder{}
	s.WriteString("\033[38;5;")
	s.WriteBytes(strconv.AppendInt(nil, colour(float64(r), float64(g), float64(b)), 10))
	s.WriteRune('m')
	s.WriteRune(c)
	s.WriteString("\033[0m")
	return s
}

func colour(red, green, blue float64) int64 {
	return 16 + baseColor(red, 36) + baseColor(green, 6) + baseColor(blue, 1)
}

func baseColor(value float64, factor int64) int64 {
	return int64(6*value/256) * factor
}
