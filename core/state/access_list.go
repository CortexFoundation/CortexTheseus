// Copyright 2020 The go-ethereum Authors
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

package state

import (
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/common"
	"maps"
	"slices"
	"strings"
)

type accessList struct {
	addresses map[common.Address]int
	slots     []map[common.Hash]struct{}
}

// ContainsAddress returns true if the address is in the access list.
func (al *accessList) ContainsAddress(address common.Address) bool {
	_, ok := al.addresses[address]
	return ok
}

// Contains checks if a slot within an account is present in the access list, returning
// separate flags for the presence of the account and the slot respectively.
func (al *accessList) Contains(address common.Address, slot common.Hash) (addressPresent bool, slotPresent bool) {
	idx, ok := al.addresses[address]
	if !ok {
		// no such address (and hence zero slots)
		return false, false
	}
	if idx == -1 {
		// address yes, but no slots
		return true, false
	}
	_, slotPresent = al.slots[idx][slot]
	return true, slotPresent
}

// newAccessList creates a new accessList.
func newAccessList() *accessList {
	return &accessList{
		addresses: make(map[common.Address]int),
	}
}

// Copy creates an independent copy of an accessList.
func (al *accessList) Copy() *accessList {
	cp := newAccessList()
	cp.addresses = maps.Clone(al.addresses)
	cp.slots = make([]map[common.Hash]struct{}, len(al.slots))
	for i, slotMap := range al.slots {
		cp.slots[i] = maps.Clone(slotMap)
	}
	return cp
}

// AddAddress adds an address to the access list, and returns 'true' if the operation
// caused a change (addr was not previously in the list).
func (al *accessList) AddAddress(address common.Address) bool {
	if _, present := al.addresses[address]; present {
		return false
	}
	al.addresses[address] = -1
	return true
}

// AddSlot adds the specified (addr, slot) combo to the access list.
// Return values are:
// - address added
// - slot added
// For any 'true' value returned, a corresponding journal entry must be made.
func (al *accessList) AddSlot(address common.Address, slot common.Hash) (addrChange bool, slotChange bool) {
	idx, addrPresent := al.addresses[address]
	if !addrPresent || idx == -1 {
		// Address not present, or addr present but no slots there
		al.addresses[address] = len(al.slots)
		slotmap := map[common.Hash]struct{}{slot: {}}
		al.slots = append(al.slots, slotmap)
		return !addrPresent, true
	}
	// There is already an (address,slot) mapping
	slotmap := al.slots[idx]
	if _, ok := slotmap[slot]; !ok {
		slotmap[slot] = struct{}{}
		// Journal add slot change
		return false, true
	}
	// No changes required
	return false, false
}

// DeleteSlot removes an (address, slot)-tuple from the access list.
// This operation needs to be performed in the same order as the addition happened.
// This method is meant to be used  by the journal, which maintains ordering of
// operations.
func (al *accessList) DeleteSlot(address common.Address, slot common.Hash) {
	idx, addrOk := al.addresses[address]
	// There are two ways this can fail
	if !addrOk {
		panic("reverting slot change, address not present in list")
	}
	slotmap := al.slots[idx]
	delete(slotmap, slot)
	// If that was the last (first) slot, remove it
	// Since additions and rollbacks are always performed in order,
	// we can delete the item without worrying about screwing up later indices
	if len(slotmap) == 0 {
		al.slots = al.slots[:idx]
		al.addresses[address] = -1
	}
}

// DeleteAddress removes an address from the access list. This operation
// needs to be performed in the same order as the addition happened.
// This method is meant to be used  by the journal, which maintains ordering of
// operations.
func (al *accessList) DeleteAddress(address common.Address) {
	delete(al.addresses, address)
}

// Equal returns true if the two access lists are identical
func (al *accessList) Equal(other *accessList) bool {
	if !maps.Equal(al.addresses, other.addresses) {
		return false
	}
	return slices.EqualFunc(al.slots, other.slots, maps.Equal)
}

// PrettyPrint prints the contents of the access list in a human-readable form
func (al *accessList) PrettyPrint() string {
	out := new(strings.Builder)
	sortedAddrs := slices.Collect(maps.Keys(al.addresses))
	slices.SortFunc(sortedAddrs, common.Address.Cmp)
	for _, addr := range sortedAddrs {
		idx := al.addresses[addr]
		fmt.Fprintf(out, "%#x : (idx %d)\n", addr, idx)
		if idx >= 0 {
			slotmap := al.slots[idx]
			for h := range slotmap {
				fmt.Fprintf(out, "    %#x\n", h)
			}
		}
	}
	return out.String()
}
