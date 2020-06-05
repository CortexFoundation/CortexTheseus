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

package core

//func TestPrintGenesisBlockHash(t *testing.T) {
//	block := DefaultGenesisBlock().ToBlock(nil)
//	t.Log(fmt.Sprintf("DefaultGenesisBlock.Hash() = %x", block.Hash()))
//	block = DefaultBernardGenesisBlock().ToBlock(nil)
//	t.Log(fmt.Sprintf("DefaultBernardGenesisBlock.Hash() = %x", block.Hash()))
//}
//
//func TestDefaultGenesisBlock(t *testing.T) {
//	block := DefaultGenesisBlock().ToBlock(nil)
//	if block.Hash() != params.MainnetGenesisHash {
//		t.Errorf("wrong mainnet genesis hash, got %v, want %v", block.Hash(), params.MainnetGenesisHash)
//	}
//	block = DefaultBernardGenesisBlock().ToBlock(nil)
//	if block.Hash() != params.BernardGenesisHash {
//		t.Errorf("wrong testnet genesis hash, got %v, want %v", block.Hash(), params.BernardGenesisHash)
//	}
//}
//
//func TestSetupGenesis(t *testing.T) {
//	var (
//		customghash = common.HexToHash("0x89c99d90b79719238d2645c7642f2c9295246e80775b38cfd162b696817fbd50")
//		customg     = Genesis{
//			Config: &params.ChainConfig{HomesteadBlock: big.NewInt(3)},
//			Alloc: GenesisAlloc{
//				{1}: {Balance: big.NewInt(1), Storage: map[common.Hash]common.Hash{{1}: {1}}},
//			},
//		}
//		oldcustomg = customg
//	)
//	oldcustomg.Config = &params.ChainConfig{HomesteadBlock: big.NewInt(2)}
//	tests := []struct {
//		name       string
//		fn         func(ctxcdb.Database) (*params.ChainConfig, common.Hash, error)
//		wantConfig *params.ChainConfig
//		wantHash   common.Hash
//		wantErr    error
//	}{
//		{
//			name: "genesis without ChainConfig",
//			fn: func(db ctxcdb.Database) (*params.ChainConfig, common.Hash, error) {
//				return SetupGenesisBlock(db, new(Genesis))
//			},
//			wantErr:    errGenesisNoConfig,
//			wantConfig: params.AllCuckooProtocolChanges,
//		},
//		{
//			name: "no block in DB, genesis == nil",
//			fn: func(db ctxcdb.Database) (*params.ChainConfig, common.Hash, error) {
//				return SetupGenesisBlock(db, nil)
//			},
//			wantHash:   params.MainnetGenesisHash,
//			wantConfig: params.MainnetChainConfig,
//		},
//		{
//			name: "mainnet block in DB, genesis == nil",
//			fn: func(db ctxcdb.Database) (*params.ChainConfig, common.Hash, error) {
//				DefaultGenesisBlock().MustCommit(db)
//				return SetupGenesisBlock(db, nil)
//			},
//			wantHash:   params.MainnetGenesisHash,
//			wantConfig: params.MainnetChainConfig,
//		},
//		{
//			name: "custom block in DB, genesis == nil",
//			fn: func(db ctxcdb.Database) (*params.ChainConfig, common.Hash, error) {
//				customg.MustCommit(db)
//				return SetupGenesisBlock(db, nil)
//			},
//			wantHash:   customghash,
//			wantConfig: customg.Config,
//		},
//		{
//			name: "custom block in DB, genesis == testnet",
//			fn: func(db ctxcdb.Database) (*params.ChainConfig, common.Hash, error) {
//				customg.MustCommit(db)
//				return SetupGenesisBlock(db, DefaultBernardGenesisBlock())
//			},
//			wantErr:    &GenesisMismatchError{Stored: customghash, New: params.BernardGenesisHash},
//			wantHash:   params.BernardGenesisHash,
//			wantConfig: params.BernardChainConfig,
//		},
//		{
//			name: "compatible config in DB",
//			fn: func(db ctxcdb.Database) (*params.ChainConfig, common.Hash, error) {
//				oldcustomg.MustCommit(db)
//				return SetupGenesisBlock(db, &customg)
//			},
//			wantHash:   customghash,
//			wantConfig: customg.Config,
//		},
//		{
//			name: "incompatible config in DB",
//			fn: func(db ctxcdb.Database) (*params.ChainConfig, common.Hash, error) {
//				// Commit the 'old' genesis block with Homestead transition at #2.
//				// Advance to block #4, past the homestead transition block of customg.
//				genesis := oldcustomg.MustCommit(db)
//
//				bc, _ := NewBlockChain(db, nil, oldcustomg.Config, cuckoo.NewFullFaker(), vm.Config{})
//				defer bc.Stop()
//
//				blocks, _ := GenerateChain(oldcustomg.Config, genesis, cuckoo.NewFaker(), db, 4, nil)
//				bc.InsertChain(blocks)
//				bc.CurrentBlock()
//				// This should return a compatibility error.
//				return SetupGenesisBlock(db, &customg)
//			},
//			wantHash:   customghash,
//			wantConfig: customg.Config,
//			wantErr: &params.ConfigCompatError{
//				What:         "Homestead fork block",
//				StoredConfig: big.NewInt(2),
//				NewConfig:    big.NewInt(3),
//				RewindTo:     1,
//			},
//		},
//	}
//
//	for _, test := range tests {
//		db := rawdb.NewMemoryDatabase()
//		config, hash, err := test.fn(db)
//		// Check the return values.
//		if !reflect.DeepEqual(err, test.wantErr) {
//			spew := spew.ConfigState{DisablePointerAddresses: true, DisableCapacities: true}
//			t.Errorf("%s: returned error %#v, want %#v", test.name, spew.NewFormatter(err), spew.NewFormatter(test.wantErr))
//		}
//		if !reflect.DeepEqual(config, test.wantConfig) {
//			t.Errorf("%s:\nreturned %v\nwant     %v", test.name, config, test.wantConfig)
//		}
//		if hash != test.wantHash {
//			t.Errorf("%s: returned hash %s, want %s", test.name, hash.Hex(), test.wantHash.Hex())
//		} else if err == nil {
//			// Check database content.
//			stored := rawdb.ReadBlock(db, test.wantHash, 0)
//			if stored.Hash() != test.wantHash {
//				t.Errorf("%s: block in DB has hash %s, want %s", test.name, stored.Hash(), test.wantHash)
//			}
//		}
//	}
//}
