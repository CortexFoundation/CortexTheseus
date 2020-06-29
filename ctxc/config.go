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
	"github.com/CortexFoundation/CortexTheseus/consensus/cuckoo"
	"github.com/CortexFoundation/CortexTheseus/core"
	"github.com/CortexFoundation/CortexTheseus/ctxc/downloader"
	"github.com/CortexFoundation/CortexTheseus/ctxc/gasprice"
	"github.com/CortexFoundation/CortexTheseus/miner"
	"github.com/CortexFoundation/CortexTheseus/params"
)

// DefaultConfig contains default settings for use on the Cortex main net.
var DefaultConfig = Config{
	SyncMode:       downloader.FullSync,
	Cuckoo:         cuckoo.Config{},
	NetworkId:      21,
	DatabaseCache:  512,
	TrieCleanCache: 256,
	TrieDirtyCache: 256,
	TrieTimeout:    60 * time.Minute,
	SnapshotCache:  256,
	Miner: miner.Config{
		GasFloor: params.MinerGasFloor,
		GasCeil:  params.MinerGasCeil,
		GasPrice: big.NewInt(params.GWei),
		Recommit: 3 * time.Second,
	},

	TxPool: core.DefaultTxPoolConfig,
	GPO: gasprice.Config{
		Blocks:     20,
		Percentile: 60,
	},
	RPCTxFeeCap: 1, // 1 ctxc
}

func init() {
	home := os.Getenv("HOME")
	if home == "" {
		if user, err := user.Current(); err == nil {
			home = user.HomeDir
		}
	}
}

//go:generate gencodec -type Config -formats toml -out gen_config.go
type Config struct {
	// The genesis block, which is inserted if the database is empty.
	// If nil, the Cortex main net block is used.
	Genesis *core.Genesis `toml:",omitempty"`

	// Protocol options
	NetworkId     uint64 // Network ID to use for selecting peers to connect to
	SyncMode      downloader.SyncMode
	DiscoveryURLs []string
	NoPruning     bool
	NoPrefetch    bool   // Whether to disable prefetching and only load state on demand
	TxLookupLimit uint64 `toml:",omitempty"` // The maximum number of blocks from head whose tx indices are reserved.

	Whitelist map[uint64]common.Hash `toml:"-"`

	// Database options
	SkipBcVersionCheck bool `toml:"-"`
	DatabaseHandles    int  `toml:"-"`
	DatabaseCache      int
	DatabaseFreezer    string
	TrieCleanCache     int
	TrieDirtyCache     int
	TrieTimeout        time.Duration
	SnapshotCache      int

	// Mining options
	Miner miner.Config

	// Mining-related options
	Coinbase         common.Address `toml:",omitempty"`
	InferDeviceType  string
	InferDeviceId    int
	InferMemoryUsage int64

	Cuckoo cuckoo.Config

	// Transaction pool options
	TxPool core.TxPoolConfig

	// Gas Price Oracle options
	GPO gasprice.Config

	EnablePreimageRecording bool
	CWASMInterpreter        string
	CVMInterpreter          string

	InferURI   string
	StorageDir string

	// Miscellaneous options
	DocRoot   string   `toml:"-"`
	RPCGasCap *big.Int `toml:",omitempty"`
	// RPCTxFeeCap is the global transaction fee(price * gaslimit) cap for
	// send-transction variants. The unit is ctxc.
	RPCTxFeeCap float64                   `toml:",omitempty"`
	Checkpoint  *params.TrustedCheckpoint `toml:",omitempty"`
	// CheckpointOracle is the configuration for checkpoint oracle.
	CheckpointOracle *params.CheckpointOracleConfig `toml:",omitempty"`
}
