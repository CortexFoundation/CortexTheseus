// Copyright 2021 The CortexTheseus Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package security

import "github.com/CortexFoundation/CortexTheseus/common"

var BadAddrs = map[common.Address]int64{
	common.HexToAddress("b251ef622230a6572c5d0ef98fdfadaa8af24890"): 3140021,
	common.HexToAddress("2f5e73677634eb2dc531785ffc306811525d8c74"): 3148935,
}

func IsBlocked(addr common.Address) (bool, int64) {
	return BadAddrs[addr] > 0, BadAddrs[addr]
}
