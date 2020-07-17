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
package params

import (
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/params"
)

const (
	PER_UPLOAD_BYTES = params.PER_UPLOAD_BYTES
	UploadGas        = params.UploadGas
)

var (
	MainnetGenesisHash = common.HexToHash("0x21d6ce908e2d1464bd74bbdbf7249845493cc1ba10460758169b978e187762c1")
	BernardGenesisHash = common.HexToHash("0x89df382cf5508d366755f5f00c16666759a3267c1c244a6524bada1901237cd3")
	DoloresGenesisHash = common.HexToHash("0xe39f1aace1c91078c97e743bd6b7a692ac215e6f9124599cdcabf0a8c7dfeae5")

	TrustedCheckpoints = map[common.Hash]*TrustedCheckpoint{
		MainnetGenesisHash: MainnetTrustedCheckpoint,
	}

	MainnetTrustedCheckpoint = &TrustedCheckpoint{
		Name:          "mainnet",
		TfsCheckPoint: 395964,

		//Ipfs:"QmahVwMfapqxW8eMhS4NGwcX7vLdPYNdvTqmuxv4Q2a7cu",
		//Infohash:"df441ba9f3543099875a58cb7bcbd04ce7fee08b",

		TfsRoot: common.HexToHash("0xe78706dbcc1f853336a31a3e3f55dcb3d0d082fb8fd4b4b273fe859d657e5dcc"),
		Skips: []Skip{
			Skip{From: 1039, To: 4468}, Skip{From: 5515, To: 10888}, Skip{From: 10888, To: 137784}, Skip{From: 137790, To: 156609}, Skip{From: 160264, To: 395088}, Skip{From: 395964, To: 1261969}, Skip{From: 1261969, To: 1966080},
		},
	}
)

type TrustedCheckpoint struct {
	Name string `json:"-"`
	//	TfsBlocks     uint64      `json:"tfsBlocks"`
	TfsCheckPoint uint64 `json:"tfsCheckPoint"`
	//	TfsFiles      uint64      `json:"tfsFiles"`
	TfsRoot common.Hash `json:"tfsRoot"`
	Skips   []Skip
}

type Skip struct {
	From uint64
	To   uint64
}
