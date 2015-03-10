package rpc

import (
	"bytes"
	"encoding/json"
	"math/big"

	"github.com/ethereum/go-ethereum/ethutil"
)

// Unmarshal state is a helper method which has the ability to decode messsages
// that use the `defaultBlock` (https://github.com/ethereum/wiki/wiki/JSON-RPC#the-default-block-parameter)
// For example a `call`: [{to: "0x....", data:"0x..."}, "latest"]. The first argument is the transaction
// message and the second one refers to the block height (or state) to which to apply this `call`.
func unmarshalState(b []byte, iface interface{}, str *string) (err error) {
	var data []json.RawMessage
	if err = json.Unmarshal(b, &data); err != nil && len(data) == 0 {
		return errDecodeArgs
	}

	if err = json.Unmarshal(data[0], iface); err != nil {
		return errDecodeArgs
	}

	// Second argument is optional (transact doesn't require it)
	if len(data) > 1 {
		if err = json.Unmarshal(data[1], str); err != nil {
			return errDecodeArgs
		}
	}

	return nil
}

type GetBlockByHashArgs struct {
	BlockHash    string
	Transactions bool
}

func (args *GetBlockByHashArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments
	}
	args.BlockHash = obj[0].(string)

	if len(obj) > 1 {
		args.Transactions = obj[1].(bool)
	}

	return nil
}

type GetBlockByNumberArgs struct {
	BlockNumber  int64
	Transactions bool
}

func (args *GetBlockByNumberArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments
	}
	args.BlockNumber = ethutil.Big(obj[0].(string)).Int64()

	if len(obj) > 1 {
		args.Transactions = obj[1].(bool)
	}

	return nil
}

type NewTxArgs struct {
	From     string
	To       string
	Value    *big.Int
	Gas      *big.Int
	GasPrice *big.Int
	Data     string

	BlockHeight string
}

func (args *NewTxArgs) UnmarshalJSON(b []byte) (err error) {
	var obj struct {
		From     string `json:"from"`
		To       string `json:"to"`
		Value    string `json:"value"`
		Gas      string `json:"gas"`
		GasPrice string `json:"gasPrice"`
		Data     string `json:"data"`
	}
	var height string
	if err = unmarshalState(b, &obj, &height); err != nil {
		return err
	}

	args.From = obj.From
	args.To = obj.To
	args.Value = ethutil.Big(obj.Value)
	args.Gas = ethutil.Big(obj.Gas)
	args.GasPrice = ethutil.Big(obj.GasPrice)
	args.Data = obj.Data
	args.BlockHeight = height

	return nil
}

type GetStorageArgs struct {
	Address     string
	BlockNumber int64
}

func (args *GetStorageArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments
	}
	args.Address = obj[0].(string)

	if len(obj) > 1 {
		if obj[1].(string) == "latest" {
			args.BlockNumber = -1
		} else {
			args.BlockNumber = ethutil.Big(obj[1].(string)).Int64()
		}
	}

	return nil
}

func (args *GetStorageArgs) requirements() error {
	if len(args.Address) == 0 {
		return NewErrorWithMessage(errArguments, "Address cannot be blank")
	}
	return nil
}

type GetStorageAtArgs struct {
	Address     string
	Key         string
	BlockNumber int64
}

func (args *GetStorageAtArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 2 {
		return errArguments
	}
	args.Address = obj[0].(string)
	args.Key = obj[1].(string)

	if len(obj) > 2 {
		if obj[2].(string) == "latest" {
			args.BlockNumber = -1
		} else {
			args.BlockNumber = ethutil.Big(obj[2].(string)).Int64()
		}
	}

	return nil
}

func (args *GetStorageAtArgs) requirements() error {
	if len(args.Address) == 0 {
		return NewErrorWithMessage(errArguments, "Address cannot be blank")
	}

	if len(args.Key) == 0 {
		return NewErrorWithMessage(errArguments, "Key cannot be blank")
	}
	return nil
}

type GetTxCountArgs struct {
	Address     string
	BlockNumber int64
}

func (args *GetTxCountArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments

	}
	args.Address = obj[0].(string)

	if len(obj) > 1 {
		if obj[1].(string) == "latest" {
			args.BlockNumber = -1
		} else {
			args.BlockNumber = ethutil.Big(obj[1].(string)).Int64()
		}
	}

	return nil
}

func (args *GetTxCountArgs) requirements() error {
	if len(args.Address) == 0 {
		return NewErrorWithMessage(errArguments, "Address cannot be blank")
	}
	return nil
}

type GetBalanceArgs struct {
	Address     string
	BlockNumber int64
}

func (args *GetBalanceArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments
	}
	args.Address = obj[0].(string)

	if len(obj) > 1 {
		if obj[1].(string) == "latest" {
			args.BlockNumber = -1
		} else {
			args.BlockNumber = ethutil.Big(obj[1].(string)).Int64()
		}
	}

	return nil
}

func (args *GetBalanceArgs) requirements() error {
	if len(args.Address) == 0 {
		return NewErrorWithMessage(errArguments, "Address cannot be blank")
	}
	return nil
}

type GetDataArgs struct {
	Address     string
	BlockNumber int64
}

func (args *GetDataArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments
	}
	args.Address = obj[0].(string)

	if len(obj) > 1 {
		if obj[1].(string) == "latest" {
			args.BlockNumber = -1
		} else {
			args.BlockNumber = ethutil.Big(obj[1].(string)).Int64()
		}
	}

	return nil
}

func (args *GetDataArgs) requirements() error {
	if len(args.Address) == 0 {
		return NewErrorWithMessage(errArguments, "Address cannot be blank")
	}
	return nil
}

type Sha3Args struct {
	Data string
}

func (args *Sha3Args) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return NewErrorWithMessage(errDecodeArgs, err.Error())
	}

	if len(obj) < 1 {
		return errArguments
	}
	args.Data = obj[0].(string)

	return nil
}

// type FilterArgs struct {
// 	FromBlock uint64
// 	ToBlock   uint64
// 	Limit     uint64
// 	Offset    uint64
// 	Address   string
// 	Topics    []string
// }

// func (args *FilterArgs) UnmarshalJSON(b []byte) (err error) {
// 	var obj []struct {
// 		FromBlock string   `json:"fromBlock"`
// 		ToBlock   string   `json:"toBlock"`
// 		Limit     string   `json:"limit"`
// 		Offset    string   `json:"offset"`
// 		Address   string   `json:"address"`
// 		Topics    []string `json:"topics"`
// 	}

// 	if err = json.Unmarshal(b, &obj); err != nil {
// 		return errDecodeArgs
// 	}

// 	if len(obj) < 1 {
// 		return errArguments
// 	}
// 	args.FromBlock = uint64(ethutil.Big(obj[0].FromBlock).Int64())
// 	args.ToBlock = uint64(ethutil.Big(obj[0].ToBlock).Int64())
// 	args.Limit = uint64(ethutil.Big(obj[0].Limit).Int64())
// 	args.Offset = uint64(ethutil.Big(obj[0].Offset).Int64())
// 	args.Address = obj[0].Address
// 	args.Topics = obj[0].Topics

// 	return nil
// }

type FilterOptions struct {
	Earliest int64
	Latest   int64
	Address  interface{}
	Topics   []interface{}
	Skip     int
	Max      int
}

func (args *FilterOptions) UnmarshalJSON(b []byte) (err error) {
	var obj []struct {
		FromBlock string        `json:"fromBlock"`
		ToBlock   string        `json:"toBlock"`
		Limit     string        `json:"limit"`
		Offset    string        `json:"offset"`
		Address   string        `json:"address"`
		Topics    []interface{} `json:"topics"`
	}

	if err = json.Unmarshal(b, &obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments
	}
	args.Earliest = int64(ethutil.Big(obj[0].FromBlock).Int64())
	args.Latest = int64(ethutil.Big(obj[0].ToBlock).Int64())
	args.Max = int(ethutil.Big(obj[0].Limit).Int64())
	args.Skip = int(ethutil.Big(obj[0].Offset).Int64())
	args.Address = obj[0].Address
	args.Topics = obj[0].Topics

	return nil
}

// type FilterChangedArgs struct {
// 	n int
// }

type DbArgs struct {
	Database string
	Key      string
	Value    string
}

func (args *DbArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 2 {
		return errArguments
	}
	args.Database = obj[0].(string)
	args.Key = obj[1].(string)

	if len(obj) > 2 {
		args.Value = obj[2].(string)
	}

	return nil
}

func (a *DbArgs) requirements() error {
	if len(a.Database) == 0 {
		return NewErrorWithMessage(errArguments, "Database cannot be blank")
	}
	if len(a.Key) == 0 {
		return NewErrorWithMessage(errArguments, "Key cannot be blank")
	}
	return nil
}

type WhisperMessageArgs struct {
	Payload  string
	To       string
	From     string
	Topic    []string
	Priority uint32
	Ttl      uint32
}

func (args *WhisperMessageArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []struct {
		Payload  string
		To       string
		From     string
		Topic    []string
		Priority string
		Ttl      string
	}

	if err = json.Unmarshal(b, &obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments
	}
	args.Payload = obj[0].Payload
	args.To = obj[0].To
	args.From = obj[0].From
	args.Topic = obj[0].Topic
	args.Priority = uint32(ethutil.Big(obj[0].Priority).Int64())
	args.Ttl = uint32(ethutil.Big(obj[0].Ttl).Int64())

	return nil
}

type CompileArgs struct {
	Source string
}

func (args *CompileArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) > 0 {
		args.Source = obj[0].(string)
	}

	return nil
}

type FilterStringArgs struct {
	Word string
}

func (args *FilterStringArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []string
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errDecodeArgs
	}

	args.Word = obj[0]

	return nil
}

type FilterIdArgs struct {
	Id int
}

func (args *FilterIdArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []string
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errDecodeArgs
	}

	args.Id = int(ethutil.Big(obj[0]).Int64())

	return nil
}

type WhisperIdentityArgs struct {
	Identity string
}

func (args *WhisperIdentityArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []string
	r := bytes.NewReader(b)
	if err := json.NewDecoder(r).Decode(&obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errDecodeArgs
	}

	args.Identity = obj[0]

	return nil
}

type WhisperFilterArgs struct {
	To     string `json:"to"`
	From   string
	Topics []string
}

func (args *WhisperFilterArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []struct {
		To     string
		From   string
		Topics []string
	}

	if err = json.Unmarshal(b, &obj); err != nil {
		return errDecodeArgs
	}

	if len(obj) < 1 {
		return errArguments
	}

	args.To = obj[0].To
	args.From = obj[0].From
	args.Topics = obj[0].Topics

	return nil
}
