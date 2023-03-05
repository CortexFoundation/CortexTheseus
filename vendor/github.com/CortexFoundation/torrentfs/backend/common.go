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

package backend

import (
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/anacrolix/torrent/bencode"
	"github.com/anacrolix/torrent/metainfo"

	"github.com/jedib0t/go-pretty/v6/progress"
	"github.com/jedib0t/go-pretty/v6/text"

	"fmt"
	"os"
	"strconv"
	"strings"
)

var messageColors = []text.Color{
	text.FgRed,
	text.FgGreen,
	text.FgYellow,
	text.FgBlue,
	text.FgMagenta,
	text.FgCyan,
	text.FgWhite,
}

func GetMessage(idx int64, units *progress.Units) string {
	var message string
	switch units {
	case &progress.UnitsBytes:
		message = fmt.Sprintf("Downloading File    #%3d", idx)
	case &progress.UnitsCurrencyDollar, &progress.UnitsCurrencyEuro, &progress.UnitsCurrencyPound:
		message = fmt.Sprintf("Transferring Amount #%3d", idx)
	default:
		message = fmt.Sprintf("Calculating Total   #%3d", idx)
	}
	return message
}

func ProgressBar(x, y int64, desc string) string {
	if y == 0 {
		return "[            ] 0%"
	}
	progress := ""
	for i := 10; i > 0; i-- {
		if int64(i) > (10*x)/y {
			progress = progress + " "
		} else {
			progress = progress + "<"
		}
	}

	prog := float64(x*100) / float64(y)
	f := strconv.FormatFloat(prog, 'f', 4, 64)
	return desc + " [ " + progress + " ] " + f + "%"
}

const (
	ScaleBarLen = 10
)

func ScaleBar(from, to, sum int) string {
	if from > to || to > sum || from > sum {
		return ""
	}
	if sum < ScaleBarLen {
		from = from * ScaleBarLen / sum
		to = to * ScaleBarLen / sum
		sum = ScaleBarLen
	}

	per := sum / ScaleBarLen

	f := from / per
	t := to / per

	bar := ""
	for i := 0; i < ScaleBarLen; i++ {
		if i > t {
			bar = bar + " "
		} else if i < f {
			bar = bar + " "
		} else {
			bar = bar + ">"
		}
	}

	prog := float64((to-from)*100) / float64(sum)
	ff := strconv.FormatFloat(prog, 'f', 2, 64)

	return "[ " + bar + " ] " + ff + "%"
}

func max(as ...int64) int64 {
	ret := as[0]
	for _, a := range as[1:] {
		if a > ret {
			ret = a
		}
	}
	return ret
}

func maxInt(as ...int) int {
	ret := as[0]
	for _, a := range as[1:] {
		if a > ret {
			ret = a
		}
	}
	return ret
}

func min(as ...int64) int64 {
	ret := as[0]
	for _, a := range as[1:] {
		if a < ret {
			ret = a
		}
	}
	return ret
}
func minInt(as ...int) int {
	ret := as[0]
	for _, a := range as[1:] {
		if a < ret {
			ret = a
		}
	}
	return ret
}

func Hash(path string) (ret string, err error) {
	info := metainfo.Info{PieceLength: 4 * 1024}
	if err = info.BuildFromFilePath(path); err != nil {
		return
	}

	var bytes []byte
	if bytes, err = bencode.Marshal(info); err == nil {
		ret = strings.ToLower(common.Address(metainfo.HashBytes(bytes)).Hex())
	}

	return
}

func IsDirectory(path string) (bool, error) {
	fileInfo, err := os.Stat(path)
	if err != nil {
		return false, err
	}

	return fileInfo.IsDir(), err
}
