package core

import (
	"encoding/json"
	"fmt"
	"math/big"
	"os"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/state"
)

/*
 * This is the special genesis block.
 */

var ZeroHash256 = make([]byte, 32)
var ZeroHash160 = make([]byte, 20)
var ZeroHash512 = make([]byte, 64)
var EmptyShaList = crypto.Sha3(ethutil.Encode([]interface{}{}))
var EmptyListRoot = crypto.Sha3(ethutil.Encode(""))

var GenesisDiff = big.NewInt(131072)

func GenesisBlock(db ethutil.Database) *types.Block {
	genesis := types.NewBlock(ZeroHash256, ZeroHash160, nil, GenesisDiff, 42, "")
	genesis.Header().Number = ethutil.Big0
	genesis.Header().GasLimit = big.NewInt(1000000)
	genesis.Header().GasUsed = ethutil.Big0
	genesis.Header().Time = 0
	genesis.Header().SeedHash = make([]byte, 32)
	genesis.Header().MixDigest = make([]byte, 32)

	genesis.Td = ethutil.Big0

	genesis.SetUncles([]*types.Header{})
	genesis.SetTransactions(types.Transactions{})
	genesis.SetReceipts(types.Receipts{})

	var accounts map[string]struct{ Balance string }
	err := json.Unmarshal(genesisData, &accounts)
	if err != nil {
		fmt.Println("enable to decode genesis json data:", err)
		os.Exit(1)
	}

	statedb := state.New(genesis.Root(), db)
	for addr, account := range accounts {
		codedAddr := ethutil.Hex2Bytes(addr)
		accountState := statedb.GetAccount(codedAddr)
		accountState.SetBalance(ethutil.Big(account.Balance))
		statedb.UpdateStateObject(accountState)
	}
	statedb.Sync()
	genesis.Header().Root = statedb.Root()

	return genesis
}

var genesisData = []byte(`{
	"0000000000000000000000000000000000000001": {"balance": "1"},
	"0000000000000000000000000000000000000002": {"balance": "1"},
	"0000000000000000000000000000000000000003": {"balance": "1"},
	"0000000000000000000000000000000000000004": {"balance": "1"},
	"dbdbdb2cbd23b783741e8d7fcf51e459b497e4a6": {"balance": "1606938044258990275541962092341162602522202993782792835301376"},
	"e4157b34ea9615cfbde6b4fda419828124b70c78": {"balance": "1606938044258990275541962092341162602522202993782792835301376"},
	"b9c015918bdaba24b4ff057a92a3873d6eb201be": {"balance": "1606938044258990275541962092341162602522202993782792835301376"},
	"6c386a4b26f73c802f34673f7248bb118f97424a": {"balance": "1606938044258990275541962092341162602522202993782792835301376"},
	"cd2a3d9f938e13cd947ec05abc7fe734df8dd826": {"balance": "1606938044258990275541962092341162602522202993782792835301376"},
	"2ef47100e0787b915105fd5e3f4ff6752079d5cb": {"balance": "1606938044258990275541962092341162602522202993782792835301376"},
	"e6716f9544a56c530d868e4bfbacb172315bdead": {"balance": "1606938044258990275541962092341162602522202993782792835301376"},
	"1a26338f0d905e295fccb71fa9ea849ffa12aaf4": {"balance": "1606938044258990275541962092341162602522202993782792835301376"}
}`)
