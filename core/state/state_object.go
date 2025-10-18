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

package state

import (
	"bytes"
	"fmt"
	"io"
	"maps"
	"math/big"
	//"sync"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/CortexFoundation/CortexTheseus/trie"
)

// hasherPool holds a pool of hashers used by state objects during concurrent
// trie updates.
//var hasherPool = sync.Pool{
//	New: func() interface{} {
//		return crypto.NewKeccakState()
//	},
//}

type Storage map[common.Hash]common.Hash

func (s Storage) Copy() Storage {
	return maps.Clone(s)
}

// stateObject represents an Cortex account which is being modified.
//
// The usage pattern is as follows:
// First you need to obtain a state object.
// Account values can be accessed and modified through the object.
// Finally, call CommitTrie to write the modified storage trie into a database.
type stateObject struct {
	address  common.Address
	addrHash common.Hash         // hash of cortex address of the account
	origin   *types.StateAccount // Account original data without any change applied, nil means it was not existent
	data     types.StateAccount
	db       *StateDB

	// DB error.
	// State objects are used by the consensus core and VM which are
	// unable to deal with database-level errors. Any error that occurs
	// during a database read is memoized here and will eventually be returned
	// by StateDB.Commit.
	dbErr error

	// Write caches.
	trie Trie   // storage trie, which becomes non-nil on first access
	code []byte // contract bytecode, which gets set when code is loaded

	originStorage  Storage // Storage entry cache to avoid duplicate reads
	pendingStorage Storage // Storage entries that need to be flushed to disk, at the end of an entire block
	dirtyStorage   Storage // Storage entries that need to be flushed to disk

	// Cache flags.
	// When an object is marked suicided it will be delete from the trie
	// during the "update" phase of the state transition.
	dirtyCode bool // true if the code was updated

	// Flag whether the account was marked as self-destructed. The self-destructed
	// account is still accessible in the scope of same transaction.
	selfDestructed bool

	// This is an EIP-6780 flag indicating whether the object is eligible for
	// self-destruct according to EIP-6780. The flag could be set either when
	// the contract is just created within the current transaction, or when the
	// object was previously existent and is being deployed as a contract within
	// the current transaction.
	newContract bool
}

// empty returns whether the account is considered empty.
func (s *stateObject) empty() bool {
	return s.data.Nonce == 0 && s.data.Balance.Sign() == 0 && bytes.Equal(s.data.CodeHash, types.EmptyCodeHash.Bytes()) && s.data.Upload.Sign() == 0 && s.data.Num.Sign() == 0
}

// newObject creates a state object.
func newObject(db *StateDB, address common.Address, acct *types.StateAccount) *stateObject {
	origin := acct
	if acct == nil {
		acct = types.NewEmptyStateAccount()
	}

	return &stateObject{
		db:             db,
		address:        address,
		addrHash:       crypto.Keccak256Hash(address[:]),
		origin:         origin,
		data:           *acct,
		originStorage:  make(Storage),
		pendingStorage: make(Storage),
		dirtyStorage:   make(Storage),
	}
}

// EncodeRLP implements rlp.Encoder.
func (c *stateObject) EncodeRLP(w io.Writer) error {
	return rlp.Encode(w, c.data)
}

// setError remembers the first non-nil error it is called with.
func (s *stateObject) setError(err error) {
	if s.dbErr == nil {
		s.dbErr = err
	}
}

func (s *stateObject) markSelfdestructed() {
	s.selfDestructed = true
}

func (s *stateObject) touch() {
	s.db.journal.touchChange(s.address)
}

// getTrie returns the associated storage trie. The trie will be opened if it's
// not loaded previously. An error will be returned if trie can't be loaded.
//
// If a new trie is opened, it will be cached within the state object to allow
// subsequent reads to expand the same trie instead of reloading from disk.
func (s *stateObject) getTrie() (Trie, error) {
	if s.trie == nil {
		tr, err := s.db.db.OpenStorageTrie(s.db.originalRoot, s.address, s.data.Root, s.db.trie)
		if err != nil {
			return nil, err
		}
		s.trie = tr
	}
	return s.trie, nil
}

// getPrefetchedTrie returns the associated trie, as populated by the prefetcher
// if it's available.
//
// Note, opposed to getTrie, this method will *NOT* blindly cache the resulting
// trie in the state object. The caller might want to do that, but it's cleaner
// to break the hidden interdependency between retrieving tries from the db or
// from the prefetcher.
func (s *stateObject) getPrefetchedTrie() Trie {
	// If there's nothing to meaningfully return, let the user figure it out by
	// pulling the trie from disk.
	if s.data.Root == types.EmptyRootHash || s.db.prefetcher == nil {
		return nil
	}
	// Attempt to retrieve the trie from the prefetcher
	return s.db.prefetcher.trie(s.addrHash, s.data.Root)
}

// GetState retrieves a value from the account storage trie.
func (s *stateObject) GetState(key common.Hash) common.Hash {
	value, _ := s.getState(key)
	return value
}

// getState retrieves a value associated with the given storage key, along with
// it's original value.
func (s *stateObject) getState(key common.Hash) (common.Hash, common.Hash) {
	origin := s.GetCommittedState(key)
	value, dirty := s.dirtyStorage[key]
	if dirty {
		return value, origin
	}
	return origin, origin
}

// GetCommittedState retrieves a value from the committed account storage trie.
func (s *stateObject) GetCommittedState(key common.Hash) common.Hash {
	if value, pending := s.pendingStorage[key]; pending {
		return value
	}
	// If we have the original value cached, return that
	if value, cached := s.originStorage[key]; cached {
		return value
	}
	// If the object was destructed in *this* block (and potentially resurrected),
	// the storage has been cleared out, and we should *not* consult the previous
	// snapshot about any storage values. The only possible alternatives are:
	//   1) resurrect happened, and new slot values were set -- those should
	//      have been handles via pendingStorage above.
	//   2) we don't have new values, and can deliver empty response back
	if _, destructed := s.db.stateObjectsDestruct[s.address]; destructed {
		return common.Hash{}
	}
	// If no live objects are available, attempt to use snapshots
	var (
		enc []byte
		err error
	)
	if s.db.snap != nil {
		start := time.Now()
		enc, err = s.db.snap.Storage(s.addrHash, crypto.Keccak256Hash(key.Bytes()))
		if metrics.EnabledExpensive {
			s.db.SnapshotStorageReads += time.Since(start)
		}
	}
	// If the snapshot is unavailable or reading from it fails, load from the database.
	if s.db.snap == nil || err != nil || len(enc) == 0 {
		start := time.Now()
		tr, e := s.getTrie()
		if e != nil {
			s.setError(err)
			return common.Hash{}
		}
		enc, err = tr.TryGet(key.Bytes())
		if metrics.EnabledExpensive {
			s.db.StorageReads += time.Since(start)
		}
		if err != nil {
			s.setError(err)
			return common.Hash{}
		}
	}
	var value common.Hash
	if len(enc) > 0 {
		_, content, _, err := rlp.Split(enc)
		if err != nil {
			s.setError(err)
		}
		value.SetBytes(content)
	}
	// Independent of where we loaded the data from, add it to the prefetcher.
	// Whilst this would be a bit weird if snapshots are disabled, but we still
	// want the trie nodes to end up in the prefetcher too, so just push through.
	if s.db.prefetcher != nil && s.data.Root != types.EmptyRootHash {
		if err = s.db.prefetcher.prefetch(s.addrHash, s.origin.Root, s.address, nil, []common.Hash{key}, true); err != nil {
			log.Error("Failed to prefetch storage slot", "addr", s.address, "key", key, "err", err)
		}
	}
	s.originStorage[key] = value
	log.Trace("Committed state", "value", value, "key", key, "addr", s.address, "s.addrHash", s.addrHash, "s.data.Upload", s.data.Upload, "s.data.Num", s.data.Num)
	return value
}

// SetState updates a value in account storage.
func (s *stateObject) SetState(key, value common.Hash) {
	// If the new value is the same as old, don't set. Otherwise, track only the
	// dirty changes, supporting reverting all of it back to no change.
	prev, origin := s.getState(key)
	if prev == value {
		return
	}
	// New value is different, update and journal the change
	s.db.journal.storageChange(s.address, key, prev, origin)
	s.setState(key, value, origin)
}

// setState updates a value in account dirty storage. The dirtiness will be
// removed if the value being set equals to the original value.
func (s *stateObject) setState(key common.Hash, value common.Hash, origin common.Hash) {
	// Storage slot is set back to its original value, undo the dirty marker
	if value == origin {
		delete(s.dirtyStorage, key)
		return
	}
	s.dirtyStorage[key] = value
}

// finalise moves all dirty storage slots into the pending area to be hashed or
// committed later. It is invoked at the end of every transaction.
func (s *stateObject) finalise() {
	slotsToPrefetch := make([]common.Hash, 0, len(s.dirtyStorage))
	for key, value := range s.dirtyStorage {
		// If the slot is different from its original value, move it into the
		// pending area to be committed at the end of the block (and prefetch
		// the pathways).
		if value != s.originStorage[key] {
			s.pendingStorage[key] = value
			slotsToPrefetch = append(slotsToPrefetch, key) // Copy needed for closure
		} else {
			// Otherwise, the slot was reverted to its original value, remove it
			// from the pending area to avoid thrashing the data structure.
			delete(s.pendingStorage, key)
		}
	}
	if s.db.prefetcher != nil && len(slotsToPrefetch) > 0 && s.data.Root != types.EmptyRootHash {
		if err := s.db.prefetcher.prefetch(s.addrHash, s.data.Root, s.address, nil, slotsToPrefetch, false); err != nil {
			log.Error("Failed to prefetch slots", "addr", s.address, "slots", len(slotsToPrefetch), "err", err)
		}
	}
	if len(s.dirtyStorage) > 0 {
		s.dirtyStorage = make(Storage)
	}
	// Revoke the flag at the end of the transaction. It finalizes the status
	// of the newly-created object as it's no longer eligible for self-destruct
	// by EIP-6780. For non-newly-created objects, it's a no-op.
	s.newContract = false
}

// updateTrie is responsible for persisting cached storage changes into the
// object's storage trie. In case the storage trie is not yet loaded, this
// function will load the trie automatically. If any issues arise during the
// loading or updating of the trie, an error will be returned. Furthermore,
// this function will return the mutated storage trie, or nil if there is no
// storage change at all.
//
// It assumes all the dirty storage slots have been finalized before.
func (s *stateObject) updateTrie() (Trie, error) {
	// Short circuit if nothing changed, don't bother with hashing anything
	if len(s.pendingStorage) == 0 {
		return s.trie, nil
	}
	// Retrieve a pretecher populated trie, or fall back to the database
	tr := s.getPrefetchedTrie()
	if tr != nil {
		// Prefetcher returned a live trie, swap it out for the current one
		s.trie = tr
	} else {
		// Fetcher not running or empty trie, fallback to the database trie
		var err error
		tr, err = s.getTrie()
		if err != nil {
			s.db.setError(err)
			return nil, err
		}
	}

	// The snapshot storage map for the object
	var (
		storage map[common.Hash][]byte
		origin  map[common.Hash][]byte
	)
	// Insert all the pending storage updates into the trie

	//hasher := hasherPool.Get().(crypto.KeccakState)
	//defer hasherPool.Put(hasher)

	// Perform trie updates before deletions.  This prevents resolution of unnecessary trie nodes
	//  in circumstances similar to the following:
	//
	// Consider nodes `A` and `B` who share the same full node parent `P` and have no other siblings.
	// During the execution of a block:
	// - `A` is deleted,
	// - `C` is created, and also shares the parent `P`.
	// If the deletion is handled first, then `P` would be left with only one child, thus collapsed
	// into a shortnode. This requires `B` to be resolved from disk.
	// Whereas if the created node is handled first, then the collapse is avoided, and `B` is not resolved.
	var (
		deletions []common.Hash
		used      = make([]common.Hash, 0, len(s.pendingStorage))
	)
	for key, value := range s.pendingStorage {
		// Skip noop changes, persist actual changes
		if value == s.originStorage[key] {
			continue
		}
		prev := s.originStorage[key]
		s.originStorage[key] = value

		var v []byte
		if (value != common.Hash{}) {
			// Encoding []byte cannot fail, ok to ignore the error.
			v, _ = rlp.EncodeToBytes(common.TrimLeftZeroes(value[:]))
			if err := tr.TryUpdate(key[:], v); err != nil {
				s.setError(err)
				return nil, err
			}
			s.db.StorageUpdated.Add(1)
		} else {
			deletions = append(deletions, key)
		}
		// Cache the mutated storage slots until commit
		if storage == nil {
			s.db.storagesLock.Lock()
			if storage = s.db.storages[s.addrHash]; storage == nil {
				storage = make(map[common.Hash][]byte)
				s.db.storages[s.addrHash] = storage
			}
			s.db.storagesLock.Unlock()
		}
		//khash := crypto.HashData(hasher, key[:])
		khash := crypto.Keccak256Hash(key[:])
		storage[khash] = v // encoded will be nil if it's deleted

		// Cache the original value of mutated storage slots
		if origin == nil {
			s.db.storagesLock.Lock()
			if origin = s.db.storagesOrigin[s.address]; origin == nil {
				origin = make(map[common.Hash][]byte)
				s.db.storagesOrigin[s.address] = origin
			}
			s.db.storagesLock.Unlock()
		}
		// Track the original value of slot only if it's mutated first time
		if _, ok := origin[khash]; !ok {
			if prev == (common.Hash{}) {
				origin[khash] = nil // nil if it was not present previously
			} else {
				// Encoding []byte cannot fail, ok to ignore the error.
				b, _ := rlp.EncodeToBytes(common.TrimLeftZeroes(prev[:]))
				origin[khash] = b
			}
		}
		// Cache the items for preloading
		used = append(used, key) // Copy needed for closure
	}
	for _, key := range deletions {
		if err := tr.TryDelete(key[:]); err != nil {
			s.setError(err)
			return nil, err
		}
		s.db.StorageDeleted.Add(1)
	}

	if s.db.prefetcher != nil {
		s.db.prefetcher.used(s.addrHash, s.data.Root, nil, used)
	}

	s.pendingStorage = make(Storage) // reset pending map
	return tr, nil
}

// UpdateRoot sets the trie root to the current root hash of
func (s *stateObject) updateRoot() {
	// If nothing changed, don't bother with hashing anything
	tr, err := s.updateTrie()
	if err != nil || tr == nil {
		return
	}
	s.data.Root = tr.Hash()
}

// commit the storage trie of the object to db.
// This updates the trie root.
func (s *stateObject) commit() error {
	if s.trie == nil {
		s.origin = s.data.Copy()
		return nil
	}
	if s.dbErr != nil {
		return s.dbErr
	}
	root, err := s.trie.Commit(nil)
	if err == nil {
		s.data.Root = root
		// Update original account data after commit
		s.origin = s.data.Copy()
	}
	return err
}

// AddBalance removes amount from c's balance.
// It is used to add funds to the destination account of a transfer.
func (c *stateObject) AddBalance(amount *big.Int) {
	// EIP158: We must check emptiness for the objects such that the account
	// clearing (0,0,0 objects) can take effect.
	if amount.Sign() == 0 {
		if c.empty() {
			c.touch()
		}

		return
	}
	c.SetBalance(new(big.Int).Add(c.Balance(), amount))
}

// SubBalance removes amount from c's balance.
// It is used to remove funds from the origin account of a transfer.
func (c *stateObject) SubBalance(amount *big.Int) {
	if amount.Sign() == 0 {
		return
	}
	c.SetBalance(new(big.Int).Sub(c.Balance(), amount))
}

func (s *stateObject) SetBalance(amount *big.Int) {
	s.db.journal.balanceChange(s.address, s.data.Balance)
	s.setBalance(amount)
}

func (s *stateObject) setBalance(amount *big.Int) {
	s.data.Balance = amount
}

// Return the gas back to the origin. Used by the Virtual machine or Closures
func (s *stateObject) ReturnGas(gas *big.Int) {}

func (s *stateObject) deepCopy(db *StateDB) *stateObject {
	obj := &stateObject{
		db:             db,
		address:        s.address,
		addrHash:       s.addrHash,
		origin:         s.origin,
		data:           s.data,
		code:           s.code,
		originStorage:  s.originStorage.Copy(),
		pendingStorage: s.pendingStorage.Copy(),
		dirtyStorage:   s.dirtyStorage.Copy(),
		dirtyCode:      s.dirtyCode,
		selfDestructed: s.selfDestructed,
		newContract:    s.newContract,
	}
	switch s.trie.(type) {
	case *trie.StateTrie:
		obj.trie = mustCopyTrie(s.trie)
	case nil:
	}
	return obj
}

//
// Attribute accessors
//

// Returns the address of the contract/account
func (s *stateObject) Address() common.Address {
	return s.address
}

// Code returns the contract code associated with this object, if any.
func (s *stateObject) Code() []byte {
	if len(s.code) != 0 {
		return s.code
	}
	if bytes.Equal(s.CodeHash(), types.EmptyCodeHash.Bytes()) {
		return nil
	}
	code, err := s.db.db.ContractCode(s.address, common.BytesToHash(s.CodeHash()))
	if err != nil {
		s.setError(fmt.Errorf("can't load code hash %x: %v", s.CodeHash(), err))
	}
	s.code = code
	return code
}

// CodeSize returns the size of the contract code associated with this object,
// or zero if none. This methos is an almost mirror of Code, but uses a cache
// inside the database to avoid loading codes seen recently.
func (s *stateObject) CodeSize() int {
	if len(s.code) != 0 {
		return len(s.code)
	}
	if bytes.Equal(s.CodeHash(), types.EmptyCodeHash.Bytes()) {
		return 0
	}
	size, err := s.db.db.ContractCodeSize(s.address, common.BytesToHash(s.CodeHash()))
	if err != nil {
		s.setError(fmt.Errorf("can't load code size %x: %v", s.CodeHash(), err))
	}
	return size
}

func (s *stateObject) SetCode(codeHash common.Hash, code []byte) {
	s.db.journal.setCode(s.address)
	s.setCode(codeHash, code)
}

func (s *stateObject) setCode(codeHash common.Hash, code []byte) {
	s.code = code
	s.data.CodeHash = codeHash[:]
	s.dirtyCode = true
}

func (s *stateObject) SetNonce(nonce uint64) {
	s.db.journal.nonceChange(s.address, s.data.Nonce)
	s.setNonce(nonce)
}

func (s *stateObject) setNonce(nonce uint64) {
	s.data.Nonce = nonce
}

func (s *stateObject) CodeHash() []byte {
	return s.data.CodeHash
}

func (s *stateObject) Balance() *big.Int {
	return s.data.Balance
}

func (s *stateObject) Nonce() uint64 {
	return s.data.Nonce
}

func (s *stateObject) Root() common.Hash {
	return s.data.Root
}

// Never called, but must be present to allow stateObject to be used
// as a vm.Account interface that also satisfies the vm.ContractRef
// interface. Interfaces are awesome.
func (s *stateObject) Value() *big.Int {
	panic("Value on stateObject should never be called")
}
