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
	progress := ""
	for i := 10; i > 0; i-- {
		if int64(i) > (10*x)/y {
			progress = progress + " "
		} else {
			progress = progress + "<"
		}
	}

	prog := float64(x*100) / float64(y)
	f := strconv.FormatFloat(prog, 'f', 2, 64)
	return "[ " + progress + " ] " + f + "% " + desc
}
