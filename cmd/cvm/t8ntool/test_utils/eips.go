// Copyright 2021 The CortexTheseus Authors
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

// This file supply temporary Eips,
// May require to update in the future
package test_utils

import (
	"fmt"
	"sort"
)

// only key is used now, so don't care about the value
// keys can check in the core/vm/eips.go:EnableEIP
var activators = map[int]bool{
	2200: true,
	1884: true,
	1344: true,
}

func ValidEip(eipNum int) bool {
	_, ok := activators[eipNum]
	return ok
}

func ActivateableEips() []string {
	var nums []string
	for k := range activators {
		nums = append(nums, fmt.Sprintf("%d", k))
	}
	sort.Strings(nums)
	return nums
}
