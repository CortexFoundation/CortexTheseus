// Copyright 2018 The go-ethereum Authors
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

import (
	"encoding/binary"
	"fmt"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"golang.org/x/crypto/sha3"
)

func newUint64(val uint64) *uint64 { return &val }

// Genesis hashes to enforce below configs on.
var (
	MainnetGenesisHash = common.HexToHash("0x21d6ce908e2d1464bd74bbdbf7249845493cc1ba10460758169b978e187762c1")
	BernardGenesisHash = common.HexToHash("0xe02ba89c73d107104af1ec647c8308968b52223e0bcefd9bbd7442b6e15b0b1b")
	DoloresGenesisHash = common.HexToHash("0xe39f1aace1c91078c97e743bd6b7a692ac215e6f9124599cdcabf0a8c7dfeae5")
)

// It should be empty for first time
var TrustedCheckpoints = map[common.Hash]*TrustedCheckpoint{
	MainnetGenesisHash: MainnetTrustedCheckpoint,
	//BernardGenesisHash: BernardTrustedCheckpoint,
	DoloresGenesisHash: DoloresTrustedCheckpoint,
}

type CheckpointOracleConfig struct {
	Address   common.Address   `json:"address"`
	Signers   []common.Address `json:"signers"`
	Threshold uint64           `json:"threshold"`
}

var CheckpointOracles = map[common.Hash]*CheckpointOracleConfig{
	MainnetGenesisHash: MainnetCheckpointOracle,
}

type TrustedCheckpoint struct {
	Name         string      `json:"-"`
	SectionIndex uint64      `json:"sectionIndex"`
	SectionHead  common.Hash `json:"sectionHead"`
}

// HashEqual returns an indicator comparing the itself hash with given one.
func (c *TrustedCheckpoint) HashEqual(hash common.Hash) bool {
	if c.Empty() {
		return hash == common.Hash{}
	}
	return c.Hash() == hash
}

// Hash returns the hash of checkpoint's four key fields(index, sectionHead, chtRoot and bloomTrieRoot).
func (c *TrustedCheckpoint) Hash() common.Hash {
	var sectionIndex [8]byte
	binary.BigEndian.PutUint64(sectionIndex[:], c.SectionIndex)

	w := sha3.NewLegacyKeccak256()
	w.Write(sectionIndex[:])
	w.Write(c.SectionHead[:])
	//w.Write(c.CHTRoot[:])
	//w.Write(c.BloomRoot[:])

	var h common.Hash
	w.Sum(h[:0])
	return h
}

// Empty returns an indicator whether the checkpoint is regarded as empty.
func (c *TrustedCheckpoint) Empty() bool {
	return c.SectionHead == (common.Hash{}) // || c.CHTRoot == (common.Hash{}) || c.BloomRoot == (common.Hash{})
}

var (
	CortexBlockRewardPeriod  = big.NewInt(8_409_600) // Halving every four years: 365 days*24 hours*60 minutes*4 blocks*4 years=8409600
	BernardBlockRewardPeriod = big.NewInt(8_409_600) // TESTING: for testnet Bernard
	DoloresBlockRewardPeriod = big.NewInt(1_000_000) // TESTING: for testnet Dolores
)

var (
	MainnetTrustedCheckpoint = &TrustedCheckpoint{
		Name:         "mainnet",
		SectionIndex: 297,
		SectionHead:  common.HexToHash("0x85a821af18c14e4449b85e048efcc9fbe8ff12627a95e31501e465319b308e50"),
	}

	DoloresTrustedCheckpoint = &TrustedCheckpoint{
		Name: "dolores",
	}

	BernardTrustedCheckpoint = &TrustedCheckpoint{
		Name: "bernard",
	}

	MainnetCheckpointOracle = &CheckpointOracleConfig{
		Address: common.HexToAddress("0x9a9070028361F7AAbeB3f2F2Dc07F82C4a98A02a"),
		Signers: []common.Address{
			//                common.HexToAddress("0x1b2C260efc720BE89101890E4Db589b44E950527"),
			//                common.HexToAddress("0x78d1aD571A1A09D60D9BBf25894b44e4C8859595"),
			//                common.HexToAddress("0x286834935f4A8Cfb4FF4C77D5770C2775aE2b0E7"),
			//                common.HexToAddress("0xb86e2B0Ab5A4B1373e40c51A7C712c70Ba2f9f8E"),
			//                common.HexToAddress("0x0DF8fa387C602AE62559cC4aFa4972A7045d6707"),
		},
		Threshold: 2,
	}
	// MainnetChainConfig is the chain parameters to run a node on the main network.
	MainnetChainConfig = &ChainConfig{
		ChainID:                       big.NewInt(21), // If all gold available to men is melted into a cube, it's approx. 21mx21mx21m
		HomesteadBlock:                big.NewInt(0),
		DAOForkBlock:                  big.NewInt(0),
		DAOForkSupport:                false,
		EIP150Block:                   big.NewInt(0), // TangerineWhistle
		EIP150Hash:                    common.HexToHash("0x"),
		EIP155Block:                   big.NewInt(0), // SpuriousDragon
		EIP158Block:                   big.NewInt(0), // SpuriousDragon
		ByzantiumBlock:                big.NewInt(0),
		ConstantinopleBlock:           big.NewInt(0),
		PetersburgBlock:               big.NewInt(0),
		IstanbulBlock:                 big.NewInt(3_230_000),
		NeoBlock:                      big.NewInt(4_650_000),
		MergeBlock:                    nil,
		OsakaTime:                     nil,
		TerminalTotalDifficulty:       nil,
		TerminalTotalDifficultyPassed: false,
		Cuckoo:                        new(CuckooConfig),
	}

	// TestnetChainConfig contains the chain parameters to run a node on the Bernard test network.
	BernardChainConfig = &ChainConfig{
		ChainID:                       big.NewInt(42),
		HomesteadBlock:                big.NewInt(0),
		DAOForkBlock:                  big.NewInt(0),
		DAOForkSupport:                false,
		EIP150Block:                   big.NewInt(0),
		EIP150Hash:                    common.HexToHash("0x"),
		EIP155Block:                   big.NewInt(0),
		EIP158Block:                   big.NewInt(0),
		ByzantiumBlock:                big.NewInt(0),
		ConstantinopleBlock:           big.NewInt(0),
		PetersburgBlock:               big.NewInt(0),
		IstanbulBlock:                 big.NewInt(0),
		NeoBlock:                      big.NewInt(5_000_000),
		MergeBlock:                    nil,
		OsakaTime:                     nil,
		TerminalTotalDifficulty:       nil,
		TerminalTotalDifficultyPassed: false,
		Clique: &CliqueConfig{
			Period: 3,
			Epoch:  30000,
		},
	}

	// TestnetChainConfig contains the chain parameters to run a node on the Dolores test network.
	DoloresChainConfig = &ChainConfig{
		ChainID:                       big.NewInt(43),
		HomesteadBlock:                big.NewInt(0),
		DAOForkBlock:                  big.NewInt(0),
		DAOForkSupport:                false,
		EIP150Block:                   big.NewInt(0),
		EIP150Hash:                    common.HexToHash("0x"),
		EIP155Block:                   big.NewInt(0),
		EIP158Block:                   big.NewInt(0),
		ByzantiumBlock:                big.NewInt(0),
		ConstantinopleBlock:           big.NewInt(0),
		PetersburgBlock:               big.NewInt(0),
		IstanbulBlock:                 big.NewInt(0),
		NeoBlock:                      big.NewInt(0),
		MergeBlock:                    nil,
		OsakaTime:                     nil,
		TerminalTotalDifficulty:       nil,
		TerminalTotalDifficultyPassed: false,
		Cuckoo:                        new(CuckooConfig),
	}

	// TestnetChainConfig contains the chain parameters to run a node on the flood test network.
	FloodChainConfig = &ChainConfig{
		ChainID:                       big.NewInt(44),
		HomesteadBlock:                big.NewInt(0),
		DAOForkBlock:                  big.NewInt(0),
		DAOForkSupport:                false,
		EIP150Block:                   big.NewInt(0),
		EIP150Hash:                    common.HexToHash("0x"),
		EIP155Block:                   big.NewInt(0),
		EIP158Block:                   big.NewInt(0),
		ByzantiumBlock:                big.NewInt(0),
		ConstantinopleBlock:           big.NewInt(0),
		PetersburgBlock:               big.NewInt(0),
		IstanbulBlock:                 big.NewInt(0),
		NeoBlock:                      big.NewInt(0),
		MergeBlock:                    nil,
		OsakaTime:                     nil,
		TerminalTotalDifficulty:       nil,
		TerminalTotalDifficultyPassed: false,
		Clique: &CliqueConfig{
			Period: 15,
			Epoch:  30000,
		},
	}

	// AllCuckooProtocolChanges contains every protocol change (EIPs) introduced
	// and accepted by the Cortex core developers into the Cuckoo consensus.
	//
	// This configuration is intentionally not using keyed fields to force anyone
	// adding flags to the config to also have to set these fields.
	AllCuckooProtocolChanges = &ChainConfig{
		ChainID:                       big.NewInt(1337),
		HomesteadBlock:                big.NewInt(0),
		DAOForkBlock:                  nil,
		DAOForkSupport:                false,
		EIP150Block:                   big.NewInt(0),
		EIP150Hash:                    common.Hash{},
		EIP155Block:                   big.NewInt(0),
		EIP158Block:                   big.NewInt(0),
		ByzantiumBlock:                nil,
		ConstantinopleBlock:           big.NewInt(0),
		PetersburgBlock:               big.NewInt(0),
		IstanbulBlock:                 nil,
		NeoBlock:                      nil,
		MergeBlock:                    nil,
		OsakaTime:                     nil,
		TerminalTotalDifficulty:       nil,
		TerminalTotalDifficultyPassed: false,
		Cuckoo:                        new(CuckooConfig),
		Clique:                        nil}

	// AllCliqueProtocolChanges contains every protocol change (EIPs) introduced
	// and accepted by the Cortex core developers into the Clique consensus.
	//
	// This configuration is intentionally not using keyed fields to force anyone
	// adding flags to the config to also have to set these fields.
	AllCliqueProtocolChanges = &ChainConfig{big.NewInt(1337), big.NewInt(0), nil, false, big.NewInt(0), common.Hash{}, big.NewInt(0), big.NewInt(0), big.NewInt(0), big.NewInt(0), nil, nil, nil, nil, nil, nil, false, nil, &CliqueConfig{Period: 0, Epoch: 30000}}

	TestChainConfig = &ChainConfig{big.NewInt(1), big.NewInt(0), nil, false, big.NewInt(0), common.Hash{}, big.NewInt(0), big.NewInt(0), big.NewInt(0), big.NewInt(0), nil, nil, nil, nil, nil, nil, false, new(CuckooConfig), nil}
	TestRules       = TestChainConfig.Rules(new(big.Int), false, 0)
)

// NetworkNames are user friendly names to use in the chain spec banner.
var NetworkNames = map[string]string{
	MainnetChainConfig.ChainID.String(): "mainnet",
	BernardChainConfig.ChainID.String(): "bernard",
	DoloresChainConfig.ChainID.String(): "dolores",
}

// ChainConfig is the core config which determines the blockchain settings.
//
// ChainConfig is stored in the database on a per block basis. This means
// that any network, identified by its genesis block, can have its own
// set of configuration options.
type ChainConfig struct {
	ChainID *big.Int `json:"chainId"` // chainId identifies the current chain and is used for replay protection

	HomesteadBlock *big.Int `json:"homesteadBlock,omitempty"` // Homestead switch block (nil = no fork, 0 = already homestead)

	DAOForkBlock   *big.Int `json:"daoForkBlock,omitempty"`   // TheDAO hard-fork switch block (nil = no fork)
	DAOForkSupport bool     `json:"daoForkSupport,omitempty"` // Whether the nodes supports or opposes the DAO hard-fork

	// EIP150 implements the Gas price changes (https://github.com/cortex/EIPs/issues/150)
	EIP150Block *big.Int    `json:"eip150Block,omitempty"` // EIP150 HF block (nil = no fork)
	EIP150Hash  common.Hash `json:"eip150Hash,omitempty"`  // EIP150 HF hash (needed for header only clients as only gas pricing changed)

	EIP155Block *big.Int `json:"eip155Block,omitempty"` // EIP155 HF block
	EIP158Block *big.Int `json:"eip158Block,omitempty"` // EIP158 HF block

	ByzantiumBlock      *big.Int `json:"byzantiumBlock,omitempty"`      // Byzantium switch block (nil = no fork, 0 = already on byzantium)
	ConstantinopleBlock *big.Int `json:"constantinopleBlock,omitempty"` // Constantinople switch block (nil = no fork, 0 = already activated)
	PetersburgBlock     *big.Int `json:"petersburgBlock,omitempty"`     // Petersburg switch block (nil = same as Constantinople)
	IstanbulBlock       *big.Int `json:"istanbulBlock,omitempty"`       // Istanbul switch block (nil = no fork, 0 = already on istanbul)
	NeoBlock            *big.Int `json:"neoBlock,omitempty"`
	MergeBlock          *big.Int `json:"mergeBlock,omitempty"`
	OsakaTime           *uint64  `json:"osakaTime,omitempty"` // Osaka switch time (nil = no fork, 0 = already on osaka)
	// TerminalTotalDifficulty is the amount of total difficulty reached by
	// the network that triggers the consensus upgrade.
	TerminalTotalDifficulty *big.Int `json:"terminalTotalDifficulty,omitempty"`

	// TerminalTotalDifficultyPassed is a flag specifying that the network already
	// passed the terminal total difficulty. Its purpose is to disable legacy sync
	// even without having seen the TTD locally (safer long term).
	TerminalTotalDifficultyPassed bool `json:"terminalTotalDifficultyPassed,omitempty"`
	// Various consensus engines
	Cuckoo *CuckooConfig `json:"cuckoo,omitempty"`
	Clique *CliqueConfig `json:"clique,omitempty"`
}

type CuckooConfig struct{}

// String implements the stringer interface, returning the consensus engine details.
func (c *CuckooConfig) String() string {
	return "cuckoo"
}

// CliqueConfig is the consensus engine configs for proof-of-authority based sealing.
type CliqueConfig struct {
	Period uint64 `json:"period"` // Number of seconds between blocks to enforce
	Epoch  uint64 `json:"epoch"`  // Epoch length to reset votes and checkpoint
}

// String implements the stringer interface, returning the consensus engine details.
func (c *CliqueConfig) String() string {
	return "clique"
}

// String implements the fmt.Stringer interface.
func (c *ChainConfig) String() string {
	var engine any
	switch {
	case c.Cuckoo != nil:
		engine = c.Cuckoo
	case c.Clique != nil:
		engine = c.Clique
	default:
		engine = "unknown"
	}
	return fmt.Sprintf("{ChainID: %v Homestead: %v DAO: %v DAOSupport: %v TangerineWhistle(EIP150): %v SpuriousDragon(EIP155): %v SpuriousDragon(EIP158): %v Byzantium: %v Constantinople: %v Petersburg: %v Istanbul: %v Neo:%v Engine: %v}",
		c.ChainID,
		c.HomesteadBlock,
		c.DAOForkBlock,
		c.DAOForkSupport,
		c.EIP150Block,
		c.EIP155Block,
		c.EIP158Block,
		c.ByzantiumBlock,
		c.ConstantinopleBlock,
		c.PetersburgBlock,
		c.IstanbulBlock,
		c.NeoBlock,
		engine,
	)
}

// IsHomestead returns whether num is either equal to the homestead block or greater.
func (c *ChainConfig) IsHomestead(num *big.Int) bool {
	return isForked(c.HomesteadBlock, num)
}

// IsDAOFork returns whether num is either equal to the DAO fork block or greater.
func (c *ChainConfig) IsDAOFork(num *big.Int) bool {
	return isForked(c.DAOForkBlock, num)
}

// IsEIP150 returns whether num is either equal to the EIP150 fork block or greater.
func (c *ChainConfig) IsEIP150(num *big.Int) bool {
	return isForked(c.EIP150Block, num)
}

// IsEIP155 returns whether num is either equal to the EIP155 fork block or greater.
func (c *ChainConfig) IsEIP155(num *big.Int) bool {
	return isForked(c.EIP155Block, num)
}

// IsEIP158 returns whether num is either equal to the EIP158 fork block or greater.
func (c *ChainConfig) IsEIP158(num *big.Int) bool {
	return isForked(c.EIP158Block, num)
}

// IsByzantium returns whether num is either equal to the Byzantium fork block or greater.
func (c *ChainConfig) IsByzantium(num *big.Int) bool {
	return isForked(c.ByzantiumBlock, num)
}

// IsConstantinople returns whether num is either equal to the Constantinople fork block or greater.
func (c *ChainConfig) IsConstantinople(num *big.Int) bool {
	return isForked(c.ConstantinopleBlock, num)
}

// IsPetersburg returns whether num is either
// - equal to or greater than the PetersburgBlock fork block,
// - OR is nil, and Constantinople is active
func (c *ChainConfig) IsPetersburg(num *big.Int) bool {
	return isForked(c.PetersburgBlock, num) || c.PetersburgBlock == nil && isForked(c.ConstantinopleBlock, num)
}

func (c *ChainConfig) IsIstanbul(num *big.Int) bool {
	return isForked(c.IstanbulBlock, num)
}

func (c *ChainConfig) IsNeo(num *big.Int) bool {
	return isForked(c.NeoBlock, num)
}

func (c *ChainConfig) IsMerge(num *big.Int) bool {
	return isForked(c.MergeBlock, num)
}

// IsTerminalPoWBlock returns whether the given block is the last block of PoW stage.
func (c *ChainConfig) IsTerminalPoWBlock(parentTotalDiff *big.Int, totalDiff *big.Int) bool {
	if c.TerminalTotalDifficulty == nil {
		return false
	}
	return parentTotalDiff.Cmp(c.TerminalTotalDifficulty) < 0 && totalDiff.Cmp(c.TerminalTotalDifficulty) >= 0
}

// GasTable returns the gas table corresponding to the current phase (homestead or homestead reprice).
//
// The returned GasTable's fields shouldn't, under any circumstances, be changed.
func (c *ChainConfig) GasTable(num *big.Int) GasTable {
	if num == nil {
		return GasTableHomestead
	}
	switch {
	case c.IsConstantinople(num):
		return GasTableConstantinople
	case c.IsEIP158(num):
		return GasTableEIP158
	case c.IsEIP150(num):
		return GasTableEIP150
	default:
		return GasTableHomestead
	}
}

// CheckCompatible checks whether scheduled fork transitions have been imported
// with a mismatching chain configuration.
func (c *ChainConfig) CheckCompatible(newcfg *ChainConfig, height uint64) *ConfigCompatError {
	bhead := new(big.Int).SetUint64(height)

	// Iterate checkCompatible to find the lowest conflict.
	var lasterr *ConfigCompatError
	for {
		err := c.checkCompatible(newcfg, bhead)
		if err == nil || (lasterr != nil && err.RewindTo == lasterr.RewindTo) {
			break
		}
		lasterr = err
		bhead.SetUint64(err.RewindTo)
	}
	return lasterr
}

// to guarantee that forks can be implemented in a different order than on official networks
func (c *ChainConfig) CheckConfigForkOrder() error {
	type fork struct {
		name     string
		block    *big.Int
		optional bool
	}
	var lastFork fork
	for _, cur := range []fork{
		{name: "homesteadBlock", block: c.HomesteadBlock},
		{name: "daoForkBlock", block: c.DAOForkBlock, optional: true},
		{name: "eip150Block", block: c.EIP150Block},
		{name: "eip155Block", block: c.EIP155Block},
		{name: "eip158Block", block: c.EIP158Block},
		{name: "byzantiumBlock", block: c.ByzantiumBlock},
		{name: "constantinopleBlock", block: c.ConstantinopleBlock},
		{name: "petersburgBlock", block: c.PetersburgBlock},
		{name: "istanbulBlock", block: c.IstanbulBlock},
		{name: "neoBlock", block: c.NeoBlock},
	} {
		if lastFork.name != "" {
			// Next one must be higher number
			if lastFork.block == nil && cur.block != nil {
				return fmt.Errorf("unsupported fork ordering: %v not enabled, but %v enabled at %v",
					lastFork.name, cur.name, cur.block)
			}
			if lastFork.block != nil && cur.block != nil {
				if lastFork.block.Cmp(cur.block) > 0 {
					return fmt.Errorf("unsupported fork ordering: %v enabled at %v, but %v enabled at %v",
						lastFork.name, lastFork.block, cur.name, cur.block)
				}
			}
		}
		if !cur.optional || cur.block != nil {
			lastFork = cur
		}
	}
	return nil
}

func (c *ChainConfig) checkCompatible(newcfg *ChainConfig, head *big.Int) *ConfigCompatError {
	if isForkIncompatible(c.HomesteadBlock, newcfg.HomesteadBlock, head) {
		return newCompatError("Homestead fork block", c.HomesteadBlock, newcfg.HomesteadBlock)
	}
	if isForkIncompatible(c.DAOForkBlock, newcfg.DAOForkBlock, head) {
		return newCompatError("DAO fork block", c.DAOForkBlock, newcfg.DAOForkBlock)
	}
	if c.IsDAOFork(head) && c.DAOForkSupport != newcfg.DAOForkSupport {
		return newCompatError("DAO fork support flag", c.DAOForkBlock, newcfg.DAOForkBlock)
	}
	if isForkIncompatible(c.EIP150Block, newcfg.EIP150Block, head) {
		return newCompatError("EIP150 fork block", c.EIP150Block, newcfg.EIP150Block)
	}
	if isForkIncompatible(c.EIP155Block, newcfg.EIP155Block, head) {
		return newCompatError("EIP155 fork block", c.EIP155Block, newcfg.EIP155Block)
	}
	if isForkIncompatible(c.EIP158Block, newcfg.EIP158Block, head) {
		return newCompatError("EIP158 fork block", c.EIP158Block, newcfg.EIP158Block)
	}
	if c.IsEIP158(head) && !configNumEqual(c.ChainID, newcfg.ChainID) {
		return newCompatError("EIP158 chain ID", c.EIP158Block, newcfg.EIP158Block)
	}
	if isForkIncompatible(c.ByzantiumBlock, newcfg.ByzantiumBlock, head) {
		return newCompatError("Byzantium fork block", c.ByzantiumBlock, newcfg.ByzantiumBlock)
	}
	if isForkIncompatible(c.ConstantinopleBlock, newcfg.ConstantinopleBlock, head) {
		return newCompatError("Constantinople fork block", c.ConstantinopleBlock, newcfg.ConstantinopleBlock)
	}
	if isForkIncompatible(c.PetersburgBlock, newcfg.PetersburgBlock, head) {
		// the only case where we allow Petersburg to be set in the past is if it is equal to Constantinople
		// mainly to satisfy fork ordering requirements which state that Petersburg fork be set if Constantinople fork is set
		if isForkIncompatible(c.ConstantinopleBlock, newcfg.PetersburgBlock, head) {
			return newCompatError("Petersburg fork block", c.PetersburgBlock, newcfg.PetersburgBlock)
		}
	}
	if isForkIncompatible(c.IstanbulBlock, newcfg.IstanbulBlock, head) {
		return newCompatError("Istanbul fork block", c.IstanbulBlock, newcfg.IstanbulBlock)
	}

	if isForkIncompatible(c.NeoBlock, newcfg.NeoBlock, head) {
		return newCompatError("Neo fork block", c.NeoBlock, newcfg.NeoBlock)
	}
	return nil
}

// isForkIncompatible returns true if a fork scheduled at s1 cannot be rescheduled to
// block s2 because head is already past the fork.
func isForkIncompatible(s1, s2, head *big.Int) bool {
	return (isForked(s1, head) || isForked(s2, head)) && !configNumEqual(s1, s2)
}

// isForked returns whether a fork scheduled at block s is active at the given head block.
// (s < head)
func isForked(s, head *big.Int) bool {
	if s == nil || head == nil {
		return false
	}
	return s.Cmp(head) <= 0
}

func configNumEqual(x, y *big.Int) bool {
	if x == nil {
		return y == nil
	}
	if y == nil {
		return x == nil
	}
	return x.Cmp(y) == 0
}

// isForkBlockIncompatible returns true if a fork scheduled at block s1 cannot be
// rescheduled to block s2 because head is already past the fork.
func isForkBlockIncompatible(s1, s2, head *big.Int) bool {
	return (isBlockForked(s1, head) || isBlockForked(s2, head)) && !configBlockEqual(s1, s2)
}

// isBlockForked returns whether a fork scheduled at block s is active at the
// given head block. Whilst this method is the same as isTimestampForked, they
// are explicitly separate for clearer reading.
func isBlockForked(s, head *big.Int) bool {
	if s == nil || head == nil {
		return false
	}
	return s.Cmp(head) <= 0
}

func configBlockEqual(x, y *big.Int) bool {
	if x == nil {
		return y == nil
	}
	if y == nil {
		return x == nil
	}
	return x.Cmp(y) == 0
}

// isForkTimestampIncompatible returns true if a fork scheduled at timestamp s1
// cannot be rescheduled to timestamp s2 because head is already past the fork.
func isForkTimestampIncompatible(s1, s2 *uint64, head uint64) bool {
	return (isTimestampForked(s1, head) || isTimestampForked(s2, head)) && !configTimestampEqual(s1, s2)
}

// isTimestampForked returns whether a fork scheduled at timestamp s is active
// at the given head timestamp. Whilst this method is the same as isBlockForked,
// they are explicitly separate for clearer reading.
func isTimestampForked(s *uint64, head uint64) bool {
	if s == nil {
		return false
	}
	return *s <= head
}

func configTimestampEqual(x, y *uint64) bool {
	if x == nil {
		return y == nil
	}
	if y == nil {
		return x == nil
	}
	return *x == *y
}

// ConfigCompatError is raised if the locally-stored blockchain is initialised with a
// ChainConfig that would alter the past.
type ConfigCompatError struct {
	What string
	// block numbers of the stored and new configurations
	StoredConfig, NewConfig *big.Int
	// the block number to which the local chain must be rewound to correct the error
	RewindTo uint64
}

func newCompatError(what string, storedblock, newblock *big.Int) *ConfigCompatError {
	var rew *big.Int
	switch {
	case storedblock == nil:
		rew = newblock
	case newblock == nil || storedblock.Cmp(newblock) < 0:
		rew = storedblock
	default:
		rew = newblock
	}
	err := &ConfigCompatError{what, storedblock, newblock, 0}
	if rew != nil && rew.Sign() > 0 {
		err.RewindTo = rew.Uint64() - 1
	}
	return err
}

func (err *ConfigCompatError) Error() string {
	return fmt.Sprintf("mismatching %s in database (have %d, want %d, rewindto %d)", err.What, err.StoredConfig, err.NewConfig, err.RewindTo)
}

// Rules wraps ChainConfig and is merely syntactic sugar or can be used for functions
// that do not have or require information about the block.
//
// Rules is a one time interface meaning that it shouldn't be used in between transition
// phases.
type Rules struct {
	ChainID                                                        *big.Int
	IsHomestead, IsEIP150, IsEIP155, IsEIP158                      bool
	IsByzantium, IsConstantinople, IsPetersburg, IsIstanbul, IsNeo bool
	IsMerge, IsOsaka                                               bool
}

// Rules ensures c's ChainID is not nil.
func (c *ChainConfig) Rules(num *big.Int, isMerge bool, timestamp uint64) Rules {
	chainID := c.ChainID
	if chainID == nil {
		chainID = new(big.Int)
	}
	return Rules{ChainID: new(big.Int).Set(chainID), IsHomestead: c.IsHomestead(num), IsEIP150: c.IsEIP150(num), IsEIP155: c.IsEIP155(num), IsEIP158: c.IsEIP158(num), IsByzantium: c.IsByzantium(num), IsPetersburg: c.IsPetersburg(num), IsIstanbul: c.IsIstanbul(num), IsNeo: c.IsNeo(num), IsMerge: c.IsMerge(num), IsOsaka: c.IsOsaka(num, timestamp)}
}

// Get Mature Block
func (c *ChainConfig) GetMatureBlock() int64 {
	if c.ChainID.Uint64() == 42 {
		return BernardMatureBlks
	}
	if c.ChainID.Uint64() == 43 {
		return DoloresMatureBlks
	}
	return MatureBlks
}

// IsOsaka returns whether time is either equal to the Osaka fork time or greater.
func (c *ChainConfig) IsOsaka(num *big.Int, time uint64) bool {
	return c.IsMerge(num) && isTimestampForked(c.OsakaTime, time)
}

// Get Block uploading quota
func (c *ChainConfig) GetBlockQuota(num *big.Int) uint64 {
	if c.ChainID.Uint64() == 42 {
		return Bernard_BLOCK_QUOTA
	}
	if c.ChainID.Uint64() == 43 {
		return Dolores_BLOCK_QUOTA
	}
	return BLOCK_QUOTA
}
