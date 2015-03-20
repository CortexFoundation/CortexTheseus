package rpc

import (
	"bytes"
	"encoding/json"
	"math/big"
	"testing"
)

func TestSha3(t *testing.T) {
	input := `["0x68656c6c6f20776f726c64"]`
	expected := "0x68656c6c6f20776f726c64"

	args := new(Sha3Args)
	json.Unmarshal([]byte(input), &args)

	if args.Data != expected {
		t.Error("got %s expected %s", input, expected)
	}
}

func TestGetBalanceArgs(t *testing.T) {
	input := `["0x407d73d8a49eeb85d32cf465507dd71d507100c1", "0x1f"]`
	expected := new(GetBalanceArgs)
	expected.Address = "0x407d73d8a49eeb85d32cf465507dd71d507100c1"
	expected.BlockNumber = 31

	args := new(GetBalanceArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if err := args.requirements(); err != nil {
		t.Error(err)
	}

	if args.Address != expected.Address {
		t.Errorf("Address should be %v but is %v", expected.Address, args.Address)
	}

	if args.BlockNumber != expected.BlockNumber {
		t.Errorf("BlockNumber should be %v but is %v", expected.BlockNumber, args.BlockNumber)
	}
}

func TestGetBalanceArgsLatest(t *testing.T) {
	input := `["0x407d73d8a49eeb85d32cf465507dd71d507100c1", "latest"]`
	expected := new(GetBalanceArgs)
	expected.Address = "0x407d73d8a49eeb85d32cf465507dd71d507100c1"
	expected.BlockNumber = -1

	args := new(GetBalanceArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if err := args.requirements(); err != nil {
		t.Error(err)
	}

	if args.Address != expected.Address {
		t.Errorf("Address should be %v but is %v", expected.Address, args.Address)
	}

	if args.BlockNumber != expected.BlockNumber {
		t.Errorf("BlockNumber should be %v but is %v", expected.BlockNumber, args.BlockNumber)
	}
}

func TestGetBalanceEmptyArgs(t *testing.T) {
	input := `[]`

	args := new(GetBalanceArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}

}

func TestGetBlockByHashArgs(t *testing.T) {
	input := `["0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331", true]`
	expected := new(GetBlockByHashArgs)
	expected.BlockHash = "0xe670ec64341771606e55d6b4ca35a1a6b75ee3d5145a99d05921026d1527331"
	expected.IncludeTxs = true

	args := new(GetBlockByHashArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if args.BlockHash != expected.BlockHash {
		t.Errorf("BlockHash should be %v but is %v", expected.BlockHash, args.BlockHash)
	}

	if args.IncludeTxs != expected.IncludeTxs {
		t.Errorf("IncludeTxs should be %v but is %v", expected.IncludeTxs, args.IncludeTxs)
	}
}

func TestGetBlockByHashEmpty(t *testing.T) {
	input := `[]`

	args := new(GetBlockByHashArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestGetBlockByNumberArgs(t *testing.T) {
	input := `["0x1b4", false]`
	expected := new(GetBlockByNumberArgs)
	expected.BlockNumber = 436
	expected.IncludeTxs = false

	args := new(GetBlockByNumberArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if args.BlockNumber != expected.BlockNumber {
		t.Errorf("BlockHash should be %v but is %v", expected.BlockNumber, args.BlockNumber)
	}

	if args.IncludeTxs != expected.IncludeTxs {
		t.Errorf("IncludeTxs should be %v but is %v", expected.IncludeTxs, args.IncludeTxs)
	}
}

func TestGetBlockByNumberEmpty(t *testing.T) {
	input := `[]`

	args := new(GetBlockByNumberArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestNewTxArgs(t *testing.T) {
	input := `[{"from": "0xb60e8dd61c5d32be8058bb8eb970870f07233155",
  "to": "0xd46e8dd67c5d32be8058bb8eb970870f072445675",
  "gas": "0x76c0",
  "gasPrice": "0x9184e72a000",
  "value": "0x9184e72a000",
  "data": "0xd46e8dd67c5d32be8d46e8dd67c5d32be8058bb8eb970870f072445675058bb8eb970870f072445675"},
  "0x10"]`
	expected := new(NewTxArgs)
	expected.From = "0xb60e8dd61c5d32be8058bb8eb970870f07233155"
	expected.To = "0xd46e8dd67c5d32be8058bb8eb970870f072445675"
	expected.Gas = big.NewInt(30400)
	expected.GasPrice = big.NewInt(10000000000000)
	expected.Value = big.NewInt(10000000000000)
	expected.Data = "0xd46e8dd67c5d32be8d46e8dd67c5d32be8058bb8eb970870f072445675058bb8eb970870f072445675"
	expected.BlockNumber = big.NewInt(16).Int64()

	args := new(NewTxArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.From != args.From {
		t.Errorf("From shoud be %#v but is %#v", expected.From, args.From)
	}

	if expected.To != args.To {
		t.Errorf("To shoud be %#v but is %#v", expected.To, args.To)
	}

	if bytes.Compare(expected.Gas.Bytes(), args.Gas.Bytes()) != 0 {
		t.Errorf("Gas shoud be %#v but is %#v", expected.Gas.Bytes(), args.Gas.Bytes())
	}

	if bytes.Compare(expected.GasPrice.Bytes(), args.GasPrice.Bytes()) != 0 {
		t.Errorf("GasPrice shoud be %#v but is %#v", expected.GasPrice, args.GasPrice)
	}

	if bytes.Compare(expected.Value.Bytes(), args.Value.Bytes()) != 0 {
		t.Errorf("Value shoud be %#v but is %#v", expected.Value, args.Value)
	}

	if expected.Data != args.Data {
		t.Errorf("Data shoud be %#v but is %#v", expected.Data, args.Data)
	}

	if expected.BlockNumber != args.BlockNumber {
		t.Errorf("BlockNumber shoud be %#v but is %#v", expected.BlockNumber, args.BlockNumber)
	}
}

func TestNewTxArgsBlockInt(t *testing.T) {
	input := `[{"from": "0xb60e8dd61c5d32be8058bb8eb970870f07233155"}, 5]`
	expected := new(NewTxArgs)
	expected.From = "0xb60e8dd61c5d32be8058bb8eb970870f07233155"
	expected.BlockNumber = big.NewInt(5).Int64()

	args := new(NewTxArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.From != args.From {
		t.Errorf("From shoud be %#v but is %#v", expected.From, args.From)
	}

	if expected.BlockNumber != args.BlockNumber {
		t.Errorf("BlockNumber shoud be %#v but is %#v", expected.BlockNumber, args.BlockNumber)
	}
}

func TestNewTxArgsEmpty(t *testing.T) {
	input := `[]`

	args := new(NewTxArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestNewTxArgsReqs(t *testing.T) {
	args := new(NewTxArgs)
	args.From = "0xb60e8dd61c5d32be8058bb8eb970870f07233155"

	err := args.requirements()
	switch err.(type) {
	case nil:
		break
	default:
		t.Errorf("Get %T", err)
	}
}

func TestNewTxArgsReqsFromBlank(t *testing.T) {
	args := new(NewTxArgs)
	args.From = ""

	err := args.requirements()
	switch err.(type) {
	case nil:
		t.Error("Expected error but didn't get one")
	case *ValidationError:
		break
	default:
		t.Error("Wrong type of error")
	}
}

func TestGetStorageArgs(t *testing.T) {
	input := `["0x407d73d8a49eeb85d32cf465507dd71d507100c1", "latest"]`
	expected := new(GetStorageArgs)
	expected.Address = "0x407d73d8a49eeb85d32cf465507dd71d507100c1"
	expected.BlockNumber = -1

	args := new(GetStorageArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if err := args.requirements(); err != nil {
		t.Error(err)
	}

	if expected.Address != args.Address {
		t.Errorf("Address shoud be %#v but is %#v", expected.Address, args.Address)
	}

	if expected.BlockNumber != args.BlockNumber {
		t.Errorf("BlockNumber shoud be %#v but is %#v", expected.BlockNumber, args.BlockNumber)
	}
}

func TestGetStorageEmptyArgs(t *testing.T) {
	input := `[]`

	args := new(GetStorageArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestGetStorageAtArgs(t *testing.T) {
	input := `["0x407d73d8a49eeb85d32cf465507dd71d507100c1", "0x0", "0x2"]`
	expected := new(GetStorageAtArgs)
	expected.Address = "0x407d73d8a49eeb85d32cf465507dd71d507100c1"
	expected.Key = "0x0"
	expected.BlockNumber = 2

	args := new(GetStorageAtArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if err := args.requirements(); err != nil {
		t.Error(err)
	}

	if expected.Address != args.Address {
		t.Errorf("Address shoud be %#v but is %#v", expected.Address, args.Address)
	}

	if expected.Key != args.Key {
		t.Errorf("Address shoud be %#v but is %#v", expected.Address, args.Address)
	}

	if expected.BlockNumber != args.BlockNumber {
		t.Errorf("BlockNumber shoud be %#v but is %#v", expected.BlockNumber, args.BlockNumber)
	}
}

func TestGetStorageAtEmptyArgs(t *testing.T) {
	input := `[]`

	args := new(GetStorageAtArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestGetTxCountArgs(t *testing.T) {
	input := `["0x407d73d8a49eeb85d32cf465507dd71d507100c1", "latest"]`
	expected := new(GetTxCountArgs)
	expected.Address = "0x407d73d8a49eeb85d32cf465507dd71d507100c1"
	expected.BlockNumber = -1

	args := new(GetTxCountArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if err := args.requirements(); err != nil {
		t.Error(err)
	}

	if expected.Address != args.Address {
		t.Errorf("Address shoud be %#v but is %#v", expected.Address, args.Address)
	}

	if expected.BlockNumber != args.BlockNumber {
		t.Errorf("BlockNumber shoud be %#v but is %#v", expected.BlockNumber, args.BlockNumber)
	}
}

func TestGetTxCountEmptyArgs(t *testing.T) {
	input := `[]`

	args := new(GetTxCountArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestGetDataArgs(t *testing.T) {
	input := `["0xd5677cf67b5aa051bb40496e68ad359eb97cfbf8", "latest"]`
	expected := new(GetDataArgs)
	expected.Address = "0xd5677cf67b5aa051bb40496e68ad359eb97cfbf8"
	expected.BlockNumber = -1

	args := new(GetDataArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if err := args.requirements(); err != nil {
		t.Error(err)
	}

	if expected.Address != args.Address {
		t.Errorf("Address shoud be %#v but is %#v", expected.Address, args.Address)
	}

	if expected.BlockNumber != args.BlockNumber {
		t.Errorf("BlockNumber shoud be %#v but is %#v", expected.BlockNumber, args.BlockNumber)
	}
}

func TestGetDataEmptyArgs(t *testing.T) {
	input := `[]`

	args := new(GetDataArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestFilterOptions(t *testing.T) {
	input := `[{
  "fromBlock": "0x1",
  "toBlock": "0x2",
  "limit": "0x3",
  "offset": "0x0",
  "address": "0xd5677cf67b5aa051bb40496e68ad359eb97cfbf8",
  "topics": ["0x12341234"]}]`
	expected := new(FilterOptions)
	expected.Earliest = 1
	expected.Latest = 2
	expected.Max = 3
	expected.Skip = 0
	expected.Address = "0xd5677cf67b5aa051bb40496e68ad359eb97cfbf8"
	// expected.Topics = []string{"0x12341234"}

	args := new(FilterOptions)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.Earliest != args.Earliest {
		t.Errorf("Earliest shoud be %#v but is %#v", expected.Earliest, args.Earliest)
	}

	if expected.Latest != args.Latest {
		t.Errorf("Latest shoud be %#v but is %#v", expected.Latest, args.Latest)
	}

	if expected.Max != args.Max {
		t.Errorf("Max shoud be %#v but is %#v", expected.Max, args.Max)
	}

	if expected.Skip != args.Skip {
		t.Errorf("Skip shoud be %#v but is %#v", expected.Skip, args.Skip)
	}

	if expected.Address != args.Address {
		t.Errorf("Address shoud be %#v but is %#v", expected.Address, args.Address)
	}

	// if expected.Topics != args.Topics {
	// 	t.Errorf("Topic shoud be %#v but is %#v", expected.Topic, args.Topic)
	// }
}

func TestFilterOptionsWords(t *testing.T) {
	input := `[{
  "fromBlock": "latest",
  "toBlock": "pending"
  }]`
	expected := new(FilterOptions)
	expected.Earliest = 0
	expected.Latest = -1

	args := new(FilterOptions)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.Earliest != args.Earliest {
		t.Errorf("Earliest shoud be %#v but is %#v", expected.Earliest, args.Earliest)
	}

	if expected.Latest != args.Latest {
		t.Errorf("Latest shoud be %#v but is %#v", expected.Latest, args.Latest)
	}
}

func TestFilterOptionsNums(t *testing.T) {
	input := `[{
  "fromBlock": 2,
  "toBlock": 3
  }]`

	args := new(FilterOptions)
	err := json.Unmarshal([]byte(input), &args)
	switch err.(type) {
	case *DecodeParamError:
		break
	default:
		t.Errorf("Should have *DecodeParamError but instead have %T", err)
	}

}

func TestFilterOptionsEmptyArgs(t *testing.T) {
	input := `[]`

	args := new(FilterOptions)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestDbArgs(t *testing.T) {
	input := `["0x74657374","0x6b6579","0x6d79537472696e67"]`
	expected := new(DbArgs)
	expected.Database = "0x74657374"
	expected.Key = "0x6b6579"
	expected.Value = "0x6d79537472696e67"

	args := new(DbArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if err := args.requirements(); err != nil {
		t.Error(err)
	}

	if expected.Database != args.Database {
		t.Errorf("Database shoud be %#v but is %#v", expected.Database, args.Database)
	}

	if expected.Key != args.Key {
		t.Errorf("Key shoud be %#v but is %#v", expected.Key, args.Key)
	}

	if expected.Value != args.Value {
		t.Errorf("Value shoud be %#v but is %#v", expected.Value, args.Value)
	}
}

func TestWhisperMessageArgs(t *testing.T) {
	input := `[{"from":"0xc931d93e97ab07fe42d923478ba2465f2",
  "topics": ["0x68656c6c6f20776f726c64"],
  "payload":"0x68656c6c6f20776f726c64",
  "ttl": "0x64",
  "priority": "0x64"}]`
	expected := new(WhisperMessageArgs)
	expected.From = "0xc931d93e97ab07fe42d923478ba2465f2"
	expected.To = ""
	expected.Payload = "0x68656c6c6f20776f726c64"
	expected.Priority = 100
	expected.Ttl = 100
	expected.Topics = []string{"0x68656c6c6f20776f726c64"}

	args := new(WhisperMessageArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.From != args.From {
		t.Errorf("From shoud be %#v but is %#v", expected.From, args.From)
	}

	if expected.To != args.To {
		t.Errorf("To shoud be %#v but is %#v", expected.To, args.To)
	}

	if expected.Payload != args.Payload {
		t.Errorf("Value shoud be %#v but is %#v", expected.Payload, args.Payload)
	}

	if expected.Ttl != args.Ttl {
		t.Errorf("Ttl shoud be %#v but is %#v", expected.Ttl, args.Ttl)
	}

	if expected.Priority != args.Priority {
		t.Errorf("Priority shoud be %#v but is %#v", expected.Priority, args.Priority)
	}

	// if expected.Topics != args.Topics {
	// 	t.Errorf("Topic shoud be %#v but is %#v", expected.Topic, args.Topic)
	// }
}

func TestFilterIdArgs(t *testing.T) {
	input := `["0x7"]`
	expected := new(FilterIdArgs)
	expected.Id = 7

	args := new(FilterIdArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.Id != args.Id {
		t.Errorf("Id shoud be %#v but is %#v", expected.Id, args.Id)
	}
}

func TestWhsiperFilterArgs(t *testing.T) {
	input := `[{"topics": ["0x68656c6c6f20776f726c64"], "to": "0x34ag445g3455b34"}]`
	expected := new(WhisperFilterArgs)
	expected.From = ""
	expected.To = "0x34ag445g3455b34"
	expected.Topics = []string{"0x68656c6c6f20776f726c64"}

	args := new(WhisperFilterArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.From != args.From {
		t.Errorf("From shoud be %#v but is %#v", expected.From, args.From)
	}

	if expected.To != args.To {
		t.Errorf("To shoud be %#v but is %#v", expected.To, args.To)
	}

	// if expected.Topics != args.Topics {
	// 	t.Errorf("Topics shoud be %#v but is %#v", expected.Topics, args.Topics)
	// }
}

func TestCompileArgs(t *testing.T) {
	input := `["contract test { function multiply(uint a) returns(uint d) {   return a * 7;   } }"]`
	expected := new(CompileArgs)
	expected.Source = `contract test { function multiply(uint a) returns(uint d) {   return a * 7;   } }`

	args := new(CompileArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.Source != args.Source {
		t.Errorf("Source shoud be %#v but is %#v", expected.Source, args.Source)
	}
}

func TestFilterStringArgs(t *testing.T) {
	input := `["pending"]`
	expected := new(FilterStringArgs)
	expected.Word = "pending"

	args := new(FilterStringArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.Word != args.Word {
		t.Errorf("Word shoud be %#v but is %#v", expected.Word, args.Word)
	}
}

func TestFilterStringEmptyArgs(t *testing.T) {
	input := `[]`

	args := new(FilterStringArgs)
	err := json.Unmarshal([]byte(input), &args)
	if err == nil {
		t.Error("Expected error but didn't get one")
	}
}

func TestWhisperIdentityArgs(t *testing.T) {
	input := `["0xc931d93e97ab07fe42d923478ba2465f283"]`
	expected := new(WhisperIdentityArgs)
	expected.Identity = "0xc931d93e97ab07fe42d923478ba2465f283"

	args := new(WhisperIdentityArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.Identity != args.Identity {
		t.Errorf("Identity shoud be %#v but is %#v", expected.Identity, args.Identity)
	}
}

func TestBlockNumIndexArgs(t *testing.T) {
	input := `["0x29a", "0x0"]`
	expected := new(BlockNumIndexArgs)
	expected.BlockNumber = 666
	expected.Index = 0

	args := new(BlockNumIndexArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.BlockNumber != args.BlockNumber {
		t.Errorf("BlockNumber shoud be %#v but is %#v", expected.BlockNumber, args.BlockNumber)
	}

	if expected.Index != args.Index {
		t.Errorf("Index shoud be %#v but is %#v", expected.Index, args.Index)
	}
}

func TestHashIndexArgs(t *testing.T) {
	input := `["0xc6ef2fc5426d6ad6fd9e2a26abeab0aa2411b7ab17f30a99d3cb96aed1d1055b", "0x1"]`
	expected := new(HashIndexArgs)
	expected.Hash = "0xc6ef2fc5426d6ad6fd9e2a26abeab0aa2411b7ab17f30a99d3cb96aed1d1055b"
	expected.Index = 1

	args := new(HashIndexArgs)
	if err := json.Unmarshal([]byte(input), &args); err != nil {
		t.Error(err)
	}

	if expected.Hash != args.Hash {
		t.Errorf("Hash shoud be %#v but is %#v", expected.Hash, args.Hash)
	}

	if expected.Index != args.Index {
		t.Errorf("Index shoud be %#v but is %#v", expected.Index, args.Index)
	}
}
