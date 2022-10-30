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

package params

import "github.com/CortexFoundation/CortexTheseus/common"

// MainnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the main Cortex network.
var MainnetBootnodes = []string{
	"enode://953226ae83e451fd78fe37bbedc55892f3e3407ae2f0c6a043ff35699fa8d45829a2490b05ce8c5de5f0b9af1b67fd5c1701eea7573a587abe0a7f8e93e5c694@35.178.69.64:40404",
	"enode://78f3feb1781f61a6a52abeaeeefeddec5b52383d0b1d0cf2a14c92ba90dc60ff010f4489b9ca3bd103ae6867cc679fea1b8558bbd515a96e828a0c7b52869bb5@47.95.215.37:40404",
}

var V5Bootnodes = []string{
	"enode://881f6548ee68e087ec157e479b96046660b96385c6fc4fe5d00d65404b725e49c4b16f85a6c1e606412f5caf82a581266b7dc1568d0cec3ea738a5589fdd3d01@47.242.33.115:37566",
}

// TestnetBootnodes are the enode URLs of the P2P bootstrap nodes running on the
// Bernard test network.
var BernardBootnodes = []string{
	"enode://cba70f4f25fa9ab33d3a92e1e5350d4e821775fc015223c77487bbe05976c7d9ec0da7f778e97ecfde208cce28e4ca7ab22b131df87419809d44ef317efc8a90@47.88.214.96:37566",
}

var DoloresBootnodes = []string{
	"enode://3bba372ec3d25442c20031064bfabb472ab7dae72d447860388797336dc35f6012c7ced2f2946508a0297d0237c7dc42d41c88e7d7ca7a82a288fa46299bda88@47.89.178.175:37566",
}

// DiscoveryV5Bootnodes are the enode URLs of the P2P bootstrap nodes for the
// experimental RLPx v5 topic-discovery network.
////var DiscoveryV5Bootnodes = []string{
//}

// const dnsPrefix = "enrtree://AKLET737XA6CY7T4QAPFJCUZRZ46EFAGZIV6LOAGKTG45RRZSUUWI@"
const dnsPrefix = "enrtree://AKLET737XA6CY7T4QAPFJCUZRZ46EFAGZIV6LOAGKTG45RRZSUUWI@"

func KnownDNSNetwork(genesis common.Hash, protocol string) string {
	var net string
	switch genesis {
	case MainnetGenesisHash:
		net = "mainnet"
	case DoloresGenesisHash:
		net = "dolores"
	case BernardGenesisHash:
		net = "bernard"
	}
	return dnsPrefix + protocol + "." + net + ".coinbag.org"
}
