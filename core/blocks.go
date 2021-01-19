// Copyright 2019 The CortexTheseus Authors
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

package core

import "github.com/CortexFoundation/CortexTheseus/common"

// BadHashes represent a set of manually tracked bad hashes (usually hard forks)
var BadHashes = map[common.Hash]bool{
	//common.HexToHash("aa12c632067adcc8acbda5dbb524687f81ece0526b9f7eb246caaadbd5ddd206"): true,
	//common.HexToHash("7d05d08cbc596a2e5e4f13b80a743e53e09221b5323c3a61946b20873e58583f"): true,
}

var BadAddrs = map[common.Address]int64{
	common.HexToAddress("b251ef622230a6572c5d0ef98fdfadaa8af24890"): 3148935,
	common.HexToAddress("2f5e73677634eb2dc531785ffc306811525d8c74"): 3148935,
}
