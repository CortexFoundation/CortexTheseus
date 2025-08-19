// Copyright 2023 The CortexTheseus Authors
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

package backend

import (
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/merkletree"
	"github.com/CortexFoundation/torrentfs/params"
	"github.com/CortexFoundation/torrentfs/types"
	bolt "go.etcd.io/bbolt"
)

func (fs *ChainDB) Leaves() []merkletree.Content {
	return fs.leaves
}

func (fs *ChainDB) initMerkleTree() error {
	if err := fs.initBlocks(); err != nil {
		return err
	}

	fs.leaves = nil
	fs.leaves = append(fs.leaves, merkletree.NewContent(params.MainnetGenesisHash.String(), uint64(0))) //BlockContent{X: params.MainnetGenesisHash.String()}) //"0x21d6ce908e2d1464bd74bbdbf7249845493cc1ba10460758169b978e187762c1"})
	tr, err := merkletree.NewTree(fs.leaves)
	if err != nil {
		return err
	}
	fs.tree = tr
	for _, block := range fs.blocks {
		if err := fs.addLeaf(block, false, false); err != nil {
			panic("Storage merkletree construct failed")
		}
	}

	log.Info("Storage merkletree initialization", "root", hexutil.Encode(fs.tree.MerkleRoot()), "number", fs.lastListenBlockNumber.Load(), "checkpoint", fs.checkPoint.Load(), "version", fs.version, "len", len(fs.blocks))

	return nil
}

func (fs *ChainDB) Metrics() time.Duration {
	return fs.treeUpdates
}

// handleNormalAdd handles the standard adding of a new leaf to the Merkle tree.
func (fs *ChainDB) handleNormalAdd(leaf *merkletree.BlockContent, number uint64) error {
	// Add the new node to the tree.
	if err := fs.tree.AddNode(leaf); err != nil {
		return fmt.Errorf("failed to add node to Merkle tree: %w", err)
	}

	// Update the checkpoint if the current block number is higher.
	if number > fs.checkPoint.Load() {
		fs.checkPoint.Store(number)
	}

	return nil
}

// handleMessing handles the special "messing" logic by rebuilding the tree from a sorted leaf list.
func (fs *ChainDB) handleMessing(leaf *merkletree.BlockContent, number uint64, dup bool) error {
	log.Debug("Messing mode activated, preparing to rebuild tree", "number", number)

	// Add the leaf if it's not a duplicate. This logic is moved here.
	if !dup {
		fs.leaves = append(fs.leaves, leaf)
	}

	// Sort the leaves by block number.
	sort.Slice(fs.leaves, func(i, j int) bool {
		return fs.leaves[i].(*merkletree.BlockContent).N() < fs.leaves[j].(*merkletree.BlockContent).N()
	})

	// Find the insertion point for the current block number.
	i := sort.Search(len(fs.leaves), func(i int) bool { return fs.leaves[i].(*merkletree.BlockContent).N() >= number })
	if i >= len(fs.leaves) {
		i = len(fs.leaves)
	}

	log.Warn("Messing solved, rebuilding tree", "number", number, "leaves_count", len(fs.leaves), "rebuild_until_index", i)

	// Rebuild the tree with the sorted and filtered leaves.
	if err := fs.tree.RebuildTreeWith(fs.leaves[0:i]); err != nil {
		return fmt.Errorf("failed to rebuild Merkle tree: %w", err)
	}

	return nil
}

func (fs *ChainDB) addLeaf(block *types.Block, mes bool, dup bool) error {
	if fs.tree == nil {
		return errors.New("mkt is nil")
	}

	number := block.Number
	leaf := merkletree.NewContent(block.Hash.String(), number)

	// Verify if the content already exists in the tree.
	inTree, verifyErr := fs.tree.VerifyContent(leaf)
	if inTree {
		log.Debug("Node is already in the tree", "num", number, "mes", mes, "dup", dup, "err", verifyErr)
		if !mes {
			// If not in messing mode, we can simply return if the node is a duplicate.
			return nil
		}
	} else {
		// If the node is not in the tree, we need to add it to the leaves list for the `mes` logic.
		// The `dup` parameter seems to control this, but the logic is a bit confusing.
		// Assuming `!dup` means it's a new leaf that needs to be appended.
		if !dup {
			fs.leaves = append(fs.leaves, leaf)
		}
	}

	// Choose the appropriate handler based on the `mes` flag.
	var err error
	if mes {
		err = fs.handleMessing(leaf, number, dup)
	} else {
		err = fs.handleNormalAdd(leaf, number)
	}

	if err != nil {
		return err
	}

	// Write the root after all operations.
	if err := fs.writeRoot(number, fs.tree.MerkleRoot()); err != nil {
		return fmt.Errorf("failed to write root: %w", err)
	}

	return nil
}

func (fs *ChainDB) Root() common.Hash {
	if fs.tree == nil {
		return common.EmptyHash
	}
	return common.BytesToHash(fs.tree.MerkleRoot())
}

func (fs *ChainDB) writeRoot(number uint64, root []byte) error {
	//fs.rootCache.Add(number, root)
	return fs.db.Update(func(tx *bolt.Tx) error {
		buk, err := tx.CreateBucketIfNotExists([]byte(VERSION_ + fs.version))
		if err != nil {
			return err
		}
		e := buk.Put([]byte(hexutil.EncodeUint64(number)), root)

		if e == nil {
			log.Debug("Root update", "number", number, "root", common.BytesToHash(root))
		}

		return e
	})
}

func (fs *ChainDB) GetRoot(number uint64) (root []byte) {
	//if root, suc := fs.rootCache.Get(number); suc {
	//	return root.([]byte)
	//}
	cb := func(tx *bolt.Tx) error {
		buk := tx.Bucket([]byte(VERSION_ + fs.version))
		if buk == nil {
			return errors.New("root bucket not exist")
		}

		v := buk.Get([]byte(hexutil.EncodeUint64(number)))
		if v == nil {
			return errors.New("root value not exist")
		}

		root = v
		return nil
	}
	if err := fs.db.View(cb); err != nil {
		return nil
	}

	return root
}
