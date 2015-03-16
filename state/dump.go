package state

import (
	"encoding/json"
	"fmt"

	"github.com/ethereum/go-ethereum/common"
)

type Account struct {
	Balance  string            `json:"balance"`
	Nonce    uint64            `json:"nonce"`
	Root     string            `json:"root"`
	CodeHash string            `json:"codeHash"`
	Storage  map[string]string `json:"storage"`
}

type World struct {
	Root     string             `json:"root"`
	Accounts map[string]Account `json:"accounts"`
}

func (self *StateDB) RawDump() World {
	world := World{
		Root:     common.Bytes2Hex(self.trie.Root()),
		Accounts: make(map[string]Account),
	}

	it := self.trie.Iterator()
	for it.Next() {
		fmt.Printf("%x\n", it.Key, len(it.Key))
		stateObject := NewStateObjectFromBytes(common.BytesToAddress(it.Key), it.Value, self.db)

		account := Account{Balance: stateObject.balance.String(), Nonce: stateObject.nonce, Root: common.Bytes2Hex(stateObject.Root()), CodeHash: common.Bytes2Hex(stateObject.codeHash)}
		account.Storage = make(map[string]string)

		storageIt := stateObject.State.trie.Iterator()
		for storageIt.Next() {
			account.Storage[common.Bytes2Hex(storageIt.Key)] = common.Bytes2Hex(storageIt.Value)
		}
		world.Accounts[common.Bytes2Hex(it.Key)] = account
	}
	return world
}

func (self *StateDB) Dump() []byte {
	json, err := json.MarshalIndent(self.RawDump(), "", "    ")
	if err != nil {
		fmt.Println("dump err", err)
	}

	return json
}

// Debug stuff
func (self *StateObject) CreateOutputForDiff() {
	fmt.Printf("%x %x %x %x\n", self.Address(), self.State.Root(), self.balance.Bytes(), self.nonce)
	it := self.State.trie.Iterator()
	for it.Next() {
		fmt.Printf("%x %x\n", it.Key, it.Value)
	}
}
