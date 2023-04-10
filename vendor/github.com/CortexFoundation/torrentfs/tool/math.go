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

package tool

import (
	"math/rand"
	"time"
)

//var smallRandTable = [16]int64{
//	0, 1, 1, 3, 0, 5, 3, 7, 3, 7, 5, 11, 7, 13, 11, 15,
//}

func Rand(s int64) int64 {
	if s == 0 {
		return 0
	}
	rand.Seed(time.Now().UnixNano())
	//	if s < int64(len(smallRandTable)) {
	//		return smallRandTable[s]
	//	}
	return rand.Int63n(s)
	//return time.Now().UnixNano() & (s - 1)
}
