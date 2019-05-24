// Copyright 2015 The go-cortex Authors
// This file is part of the go-cortex library.
//
// The go-cortex library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-cortex library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-cortex library. If not, see <http://www.gnu.org/licenses/>.

package params

// MainnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the main Cortex network.
var MainnetTrackers = []string{
	"http://47.91.91.217:5008/announce",
	"http://47.74.1.234:5008/announce",
	"http://47.88.7.24:5008/announce",
	"http://47.91.43.70:5008/announce",
	"http://47.91.106.117:5008/announce",
	"http://47.91.147.37:5008/announce",
	"http://47.89.178.175:5008/announce",
	"http://47.88.214.96:5008/announce",
}

var CerebroTrackers = []string{}

var TestnetTrackers = []string{}

// RinkebyBootnodes are the enode URLs of the P2P bootstrap nodes running on the
// Rinkeby test network.
var RinkebyTrackerss = []string{}
