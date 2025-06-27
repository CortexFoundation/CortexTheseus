// Copyright 2015 The go-ethereum Authors
// This file is part of The go-ethereum library.
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
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package trie

import (
	"fmt"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

// StateTrie is the old name of StateTrie.
// Deprecated: use StateTrie.
type SecureTrie = StateTrie

// NewSecure creates a new StateTrie.
// Deprecated: use NewStateTrie.
func NewSecure(stateRoot common.Hash, owner common.Hash, root common.Hash, db *Database) (*SecureTrie, error) {
	id := &ID{
		StateRoot: stateRoot,
		Owner:     owner,
		Root:      root,
	}
	return NewStateTrie(id, db)
}

// StateTrie wraps a trie with key hashing. In a secure trie, all
// access operations hash the key using keccak256. This prevents
// calling code from creating long chains of nodes that
// increase the access time.
//
// Contrary to a regular trie, a StateTrie can only be created with
// New and must have an attached database. The database also stores
// the preimage of each key.
//
// StateTrie is not safe for concurrent use.
type StateTrie struct {
	trie        Trie
	preimages   *preimageStore
	secKeyCache map[common.Hash][]byte
}

// NewSecure creates a trie with an existing root node from a backing database
// and optional intermediate in-memory node pool.
//
// If root is the zero hash or the sha3 hash of an empty string, the
// trie is initially empty. Otherwise, New will panic if db is nil
// and returns MissingNodeError if the root node cannot be found.
//
// Accessing the trie loads nodes from the database or node pool on demand.
// Loaded nodes are kept around until their 'cache generation' expires.
// A new cache generation is created by each call to Commit.
// cachelimit sets the number of past cache generations to keep.
func NewStateTrie(id *ID, db *Database) (*StateTrie, error) {
	if db == nil {
		panic("trie.NewSecure called without a database")
	}
	trie, err := New(id, db)
	if err != nil {
		return nil, err
	}

	tr := &StateTrie{
		trie:        *trie,
		secKeyCache: make(map[common.Hash][]byte),
	}
	if db.PreimageEnabled() {
		tr.preimages = db.preimages
	}
	return tr, nil
}

// Get returns the value for key stored in the trie.
// The value bytes must not be modified by the caller.
func (t *StateTrie) Get(key []byte) []byte {
	res, err := t.TryGet(key)
	if err != nil {
		log.Error(fmt.Sprintf("Unhandled trie error: %v", err))
	}
	return res
}

// GetStorage attempts to retrieve a storage slot with provided account address
// and slot key. The value bytes must not be modified by the caller.
// If the specified storage slot is not in the trie, nil will be returned.
// If a trie node is not found in the database, a MissingNodeError is returned.
func (t *StateTrie) GetStorage(_ common.Address, key []byte) ([]byte, error) {
	enc, err := t.TryGet(key)
	if len(enc) == 0 || err != nil {
		return nil, err
	}
	_, content, _, err := rlp.Split(enc)
	return content, err
}

// GetAccount attempts to retrieve an account with provided account address.
// If the specified account is not in the trie, nil will be returned.
// If a trie node is not found in the database, a MissingNodeError is returned.
func (t *StateTrie) GetAccount(address common.Address) (*types.StateAccount, error) {
	res, err := t.TryGet(crypto.Keccak256(address.Bytes()))
	if res == nil || err != nil {
		return nil, err
	}
	ret := new(types.StateAccount)
	err = rlp.DecodeBytes(res, ret)
	return ret, err
}

// TryGet returns the value for key stored in the trie.
// The value bytes must not be modified by the caller.
// If a node was not found in the database, a MissingNodeError is returned.
func (t *StateTrie) TryGet(key []byte) ([]byte, error) {
	return t.trie.TryGet(crypto.Keccak256(key))
}

// TryGetNode attempts to retrieve a trie node by compact-encoded path. It is not
// possible to use keybyte-encoding as the path might contain odd nibbles.
func (t *StateTrie) TryGetNode(path []byte) ([]byte, int, error) {
	return t.trie.TryGetNode(path)
}

// UpdateStorage associates key with value in the trie. Subsequent calls to
// Get will return value. If value has length zero, any existing value
// is deleted from the trie and calls to Get will return nil.
//
// The value bytes must not be modified by the caller while they are
// stored in the trie.
//
// If a node is not found in the database, a MissingNodeError is returned.
func (t *StateTrie) UpdateStorage(_ common.Address, key, value []byte) error {
	hk := crypto.Keccak256(key)
	v, _ := rlp.EncodeToBytes(value)
	err := t.trie.TryUpdate(hk, v)
	if err != nil {
		return err
	}
	if t.preimages != nil {
		t.secKeyCache[common.Hash(hk)] = common.CopyBytes(key)
	}
	return nil
}

// TryUpdateAccount account will abstract the write of an account to the
// secure trie.
func (t *StateTrie) TryUpdateAccount(key common.Address, acc *types.StateAccount) error {
	hk := crypto.Keccak256(key.Bytes())
	data, err := rlp.EncodeToBytes(acc)
	if err != nil {
		return err
	}
	if err := t.trie.TryUpdate(hk, data); err != nil {
		return err
	}
	if t.preimages != nil {
		t.secKeyCache[common.Hash(hk)] = common.CopyBytes(key.Bytes())
	}
	return nil
}

func (t *StateTrie) UpdateContractCode(_ common.Address, _ common.Hash, _ []byte) error {
	return nil
}

// Update associates key with value in the trie. Subsequent calls to
// Get will return value. If value has length zero, any existing value
// is deleted from the trie and calls to Get will return nil.
//
// The value bytes must not be modified by the caller while they are
// stored in the trie.
func (t *StateTrie) Update(key, value []byte) {
	if err := t.TryUpdate(key, value); err != nil {
		log.Error(fmt.Sprintf("Unhandled trie error: %v", err))
	}
}

// TryUpdate associates key with value in the trie. Subsequent calls to
// Get will return value. If value has length zero, any existing value
// is deleted from the trie and calls to Get will return nil.
//
// The value bytes must not be modified by the caller while they are
// stored in the trie.
//
// If a node was not found in the database, a MissingNodeError is returned.
func (t *StateTrie) TryUpdate(key, value []byte) error {
	hk := crypto.Keccak256(key)
	err := t.trie.TryUpdate(hk, value)
	if err != nil {
		return err
	}
	if t.preimages != nil {
		t.secKeyCache[common.Hash(hk)] = common.CopyBytes(key)
	}
	return nil
}

// Delete removes any existing value for key from the trie.
func (t *StateTrie) Delete(key []byte) {
	if err := t.TryDelete(key); err != nil {
		log.Error(fmt.Sprintf("Unhandled trie error: %v", err))
	}
}

// TryDelete removes any existing value for key from the trie.
// If a node was not found in the database, a MissingNodeError is returned.
func (t *StateTrie) TryDelete(key []byte) error {
	hk := crypto.Keccak256(key)
	if t.preimages != nil {
		delete(t.secKeyCache, common.Hash(hk))
	}
	return t.trie.TryDelete(hk)
}

// TryDeleteAccount abstracts an account deletion from the trie.
func (t *StateTrie) TryDeleteAccount(key []byte) error {
	hk := crypto.Keccak256(key)
	if t.preimages != nil {
		delete(t.secKeyCache, common.Hash(hk))
	}
	return t.trie.TryDelete(hk)
}

// GetKey returns the sha3 preimage of a hashed key that was
// previously used to store a value.
func (t *StateTrie) GetKey(shaKey []byte) []byte {
	if t.preimages == nil {
		return nil
	}
	if key, ok := t.secKeyCache[common.Hash(shaKey)]; ok {
		return key
	}
	return t.preimages.preimage(common.BytesToHash(shaKey))
}

// Commit writes all nodes and the secure hash pre-images to the trie's database.
// Nodes are stored with their sha3 hash as the key.
//
// Committing flushes nodes from memory. Subsequent Get calls will load nodes
// from the database.
func (t *StateTrie) Commit(onleaf LeafCallback) (root common.Hash, err error) {
	// Write all the pre-images to the actual disk database
	if len(t.secKeyCache) > 0 {
		if t.preimages != nil {
			t.preimages.insertPreimage(t.secKeyCache)
		}
		clear(t.secKeyCache)
	}
	// Commit the trie to its intermediate node database
	return t.trie.Commit(onleaf)
}

// Hash returns the root hash of StateTrie. It does not write to the
// database and can be used even if the trie doesn't have one.
func (t *StateTrie) Hash() common.Hash {
	return t.trie.Hash()
}

// Copy returns a copy of StateTrie.
func (t *StateTrie) Copy() *StateTrie {
	return &StateTrie{
		trie:        *t.trie.Copy(),
		preimages:   t.preimages,
		secKeyCache: make(map[common.Hash][]byte),
	}
}

// NodeIterator returns an iterator that returns nodes of the underlying trie. Iteration
// starts at the key after the given start key.
func (t *StateTrie) NodeIterator(start []byte) NodeIterator {
	return t.trie.NodeIterator(start)
}

// MustNodeIterator is a wrapper of NodeIterator and will omit any encountered
// error but just print out an error message.
func (t *StateTrie) MustNodeIterator(start []byte) NodeIterator {
	return t.trie.MustNodeIterator(start)
}
