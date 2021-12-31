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

package torrentfs

import (
	"strconv"
)

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
	return "[ " + progress + " ] " + f + "% " + desc
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
