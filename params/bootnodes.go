// Copyright 2015 The go-ethereum Authors
// This file is part of the go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package params

// MainnetBootnodes are the enode URLs of the P2P bootstrap nodes running on
// the main Ethereum network.
var MainnetBootnodes = []string{}

// TestnetBootnodes are the enode URLs of the P2P bootstrap nodes running on the
// Cerebro test network.
var CerebroBootnodes = []string{
	"enode://d73a0e572ae4a710a228ca39f5cdd83c366fee7cbee25249eb354463cbfb3b07f85d1b5382bdbde5d26dc8c11c6240cc4f80c5c57e71b760c39797185da0f123@47.74.15.143:37566",
	"enode://0d6eedee4ed1c4a1b9f273de9ed28fb30581b3bad6b733609870b39ed5a716cc59c5067deb1a43570b15f5965ab22ac04c7067a79aef1e240667d78e8ffdf81d@47.254.135.53:37566",
	"enode://c8119e731b29bed3222ee5e63ec514926fa683f0bc999c73ff171d54a3100aea4211d70640bf937d1d61d39fdf313470b131213ae11ddfea5ccb96669f603749@47.88.174.57:37566",
	"enode://a4d20d02a05c3674791f1a0cf9900f0db50485037f96c99395287e931a6845f624e16adbce9e84e7106f803278360979e6a6999b7ca8c25cf64b4445a6de75e6@47.52.39.170:37566",
	"enode://67ef167c2620b9c166161f7226a7b8f8d32e5282244a4ceebf6d18be70551a9b62c20f0f5aeb9b7abcbbfa062c3d925ada47e0e27942cfd1fc2cf2617f42297e@47.75.211.148:37566",
	"enode://0778ac6fe1a572a2f5834452c1d4b6f77b57a2a4baa72e0e2582ad0dbfe88f41e4e66aa020a451a32936a86c29788d257ccaa19f58e5676943595b945e615b9e@139.224.132.6:37566",
}

// RinkebyBootnodes are the enode URLs of the P2P bootstrap nodes running on the
// Rinkeby test network.
var RinkebyBootnodes = []string{}

// DiscoveryV5Bootnodes are the enode URLs of the P2P bootstrap nodes for the
// experimental RLPx v5 topic-discovery network.
var DiscoveryV5Bootnodes = []string{}
