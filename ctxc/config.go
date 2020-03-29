// Copyright 2018 The CortexTheseus Authors
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
	SyncMode:      downloader.FullSync,
	Cuckoo:        cuckoo.Config{},
	NetworkId:     21,
	DatabaseCache: 768,
	TrieCache:     256,
	TrieTimeout:   60 * time.Minute,
	SnapshotCache: 256,
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
}

//go:generate gencodec -type Config -field-override configMarshaling -formats toml -out gen_config.go

type Config struct {
	// The genesis block, which is inserted if the database is empty.
	// If nil, the Cortex main net block is used.
	Genesis *core.Genesis `toml:",omitempty"`

	// Protocol options
	NetworkId uint64 // Network ID to use for selecting peers to connect to
	SyncMode  downloader.SyncMode
	DiscoveryURLs []string
	NoPruning bool

	Whitelist map[uint64]common.Hash `toml:"-"`

	// Database options
	SkipBcVersionCheck bool `toml:"-"`
	DatabaseHandles    int  `toml:"-"`
	DatabaseCache      int
	DatabaseFreezer    string
	TrieCache          int
	TrieTimeout        time.Duration
	SnapshotCache      int

	// Mining-related options
	Coinbase         common.Address `toml:",omitempty"`
	MinerNotify      []string       `toml:",omitempty"`
	MinerExtraData   []byte         `toml:",omitempty"`
	MinerGasFloor    uint64
	MinerGasCeil     uint64
	MinerGasPrice    *big.Int
	MinerRecommit    time.Duration
	MinerNoverify    bool
	MinerCuda        bool
	MinerOpenCL      bool
	MinerDevices     string
	InferDeviceType  string
	InferDeviceId    int
	InferMemoryUsage int64

	Cuckoo cuckoo.Config

	// Transaction pool options
	TxPool core.TxPoolConfig

	// Gas Price Oracle options
	GPO gasprice.Config

	// Enables tracking of SHA3 preimages in the VM
	EnablePreimageRecording bool

	InferURI   string
	StorageDir string

	// Miscellaneous options
	DocRoot    string                    `toml:"-"`
	Checkpoint *params.TrustedCheckpoint `toml:",omitempty"`
	// CheckpointOracle is the configuration for checkpoint oracle.
	CheckpointOracle *params.CheckpointOracleConfig `toml:",omitempty"`

	// Istanbul block override (TODO: remove after the fork)
	OverrideIstanbul *big.Int `toml:",omitempty"`
}

type configMarshaling struct {
	MinerExtraData hexutil.Bytes
}
