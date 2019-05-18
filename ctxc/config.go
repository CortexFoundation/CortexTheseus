// Copyright 2017 The go-cortex Authors
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

package ctxc

import (
	"math/big"
	"os"
	"os/user"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/consensus/cuckoo"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/ctxc/downloader"
	"github.com/CortexFoundation/CortexTheseus/ctxc/gasprice"
	"github.com/CortexFoundation/CortexTheseus/params"
)

// DefaultConfig contains default settings for use on the Cortex main net.
var DefaultConfig = Config{
	SyncMode: downloader.FullSync,
	Ethash:   cuckoo.Config{
		/* CacheDir:       "ethash",
		CachesInMem:    2,
		CachesOnDisk:   3,
		DatasetsInMem:  1,
		DatasetsOnDisk: 2, */
	},
	NetworkId:     42,
	DatabaseCache: 768,
	TrieCache:     256,
	TrieTimeout:   60 * time.Minute,
	MinerGasFloor: params.MinerGasFloor, //8000000,
	MinerGasCeil:  params.MinerGasCeil,  //8000000,
	MinerGasPrice: big.NewInt(params.GWei),
	MinerRecommit: 3 * time.Second,

	TxPool: core.DefaultTxPoolConfig,
	GPO: gasprice.Config{
		Blocks:     20,
		Percentile: 60,
	},
}

func init() {
	home := os.Getenv("HOME")
	if home == "" {
		if user, err := user.Current(); err == nil {
			home = user.HomeDir
		}
	}
	/* if runtime.GOOS == "windows" {
		DefaultConfig.Ethash.DatasetDir = filepath.Join(home, "AppData", "Ethash")
	} else {
		DefaultConfig.Ethash.DatasetDir = filepath.Join(home, ".ctxcash")
	} */
}

//go:generate gencodec -type Config -field-override configMarshaling -formats toml -out gen_config.go

type Config struct {
	// The genesis block, which is inserted if the database is empty.
	// If nil, the Cortex main net block is used.
	Genesis *core.Genesis `toml:",omitempty"`

	// Protocol options
	NetworkId uint64 // Network ID to use for selecting peers to connect to
	SyncMode  downloader.SyncMode
	NoPruning bool

	Whitelist map[uint64]common.Hash `toml:"-"`

	// Database options
	SkipBcVersionCheck bool `toml:"-"`
	DatabaseHandles    int  `toml:"-"`
	DatabaseCache      int
	TrieCache          int
	TrieTimeout        time.Duration

	// Mining-related options
	Coinbase      common.Address `toml:",omitempty"`
	MinerNotify    []string       `toml:",omitempty"`
	MinerExtraData []byte         `toml:",omitempty"`
	MinerGasFloor  uint64
	MinerGasCeil   uint64
	MinerGasPrice  *big.Int
	MinerRecommit  time.Duration
	MinerNoverify  bool
	MinerCuda      bool
	MinerOpenCL    bool
	MinerDevices   string

	// Ethash options
	Ethash cuckoo.Config

	// Transaction pool options
	TxPool core.TxPoolConfig

	// Gas Price Oracle options
	GPO gasprice.Config

	// Enables tracking of SHA3 preimages in the VM
	EnablePreimageRecording bool

	InferURI   string
	StorageDir string

	// Miscellaneous options
	DocRoot string `toml:"-"`
}

type configMarshaling struct {
	MinerExtraData hexutil.Bytes
}
