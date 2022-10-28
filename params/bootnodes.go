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
	//"enode://72f70e8e77f68f5526d730f1003e04caa0995eb5c8fb66644da72b23656dc8c16075b0f740d687a3c2aa9d0db1337bf826715c69ae2adb64dc106d3d4165fd79@47.91.91.217:37566",
	//"enode://c5780febab5e5a7bd6387a20a2662df3f1b16a10d93931a40a147e0f6cfd89a576c2e2f758e0e886c3f91a1bc43b3c7fa01af0c8b8ce39c8004c048ca880bccf@47.74.1.234:37566",
	//"enode://9b3b10d4223e010b01411a252312fb69da63b88fd610c07adb5bfa941a8598009a4bb2deeac42c41498acbdaec2196e2cc1fe746286c46f0b5c47d42c5c777b3@47.88.7.24:37566",
	//"enode://8a379f3aa5d6b35ea9b594252f092f416ea583f3d2fbf7494db2fcfffc91fda9fdd38b34c2e6b6ff003d03ecf67c67326388a53b953361307daf956902c2187f@47.91.43.70:37566",
	//"enode://6c284d3f0bcbbce419ebdee0510c3cd07207edf1d70de3a3e75c5bcb66c84c56d5eec7caec191770173a9a995d3592469cb9052ef65e2da775136006a9ff9c79@47.91.106.117:37566",
	//"enode://838b42c2f6532750c36ac346f66109ac8d904a7e4271a1e999d0909f6563591cfa796cc00af77b7d8bdf8643507cb7afe71a8da7e7c4d66a1a93018eee88f6f7@47.91.147.37:37566",
	//"enode://3bba372ec3d25442c20031064bfabb472ab7dae72d447860388797336dc35f6012c7ced2f2946508a0297d0237c7dc42d41c88e7d7ca7a82a288fa46299bda88@47.89.178.175:37566",
	//"enode://cba70f4f25fa9ab33d3a92e1e5350d4e821775fc015223c77487bbe05976c7d9ec0da7f778e97ecfde208cce28e4ca7ab22b131df87419809d44ef317efc8a90@47.88.214.96:37566",
	//"enode://3ed8bf8513bc6e75d1077958ebb6fc4c69bde8decff95abf88b3bfb0c22584d6b467d1b0f43a8de2c1d9bf4bf5ab57f12232363dae271272c75fa3227a7b1b3a@47.242.33.115:37566",
}

var V5Bootnodes = []string{
	//"enode://838b42c2f6532750c36ac346f66109ac8d904a7e4271a1e999d0909f6563591cfa796cc00af77b7d8bdf8643507cb7afe71a8da7e7c4d66a1a93018eee88f6f7@47.91.147.37:37566",
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
