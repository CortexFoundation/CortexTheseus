package rpc

import (
	"encoding/json"
	"fmt"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
)

const (
	defaultLogLimit  = 100
	defaultLogOffset = 0
)

func blockHeightFromJson(msg json.RawMessage, number *int64) error {
	var raw interface{}
	if err := json.Unmarshal(msg, &raw); err != nil {
		return NewDecodeParamError(err.Error())
	}
	return blockHeight(raw, number)
}

func blockHeight(raw interface{}, number *int64) error {
	// Parse as integer
	num, ok := raw.(float64)
	if ok {
		*number = int64(num)
		return nil
	}

	// Parse as string/hexstring
	str, ok := raw.(string)
	if !ok {
		return NewInvalidTypeError("", "not a number or string")
	}

	switch str {
	case "latest":
		*number = -1
	case "pending":
		*number = -2
	default:
		*number = common.String2Big(str).Int64()
	}

	return nil
}

func numString(raw interface{}, number *int64) error {
	// Parse as integer
	num, ok := raw.(float64)
	if ok {
		*number = int64(num)
		return nil
	}

	// Parse as string/hexstring
	str, ok := raw.(string)
	if !ok {
		return NewInvalidTypeError("", "not a number or string")
	}
	*number = common.String2Big(str).Int64()

	return nil
}

// func toNumber(v interface{}) (int64, error) {
// 	var str string
// 	if v != nil {
// 		var ok bool
// 		str, ok = v.(string)
// 		if !ok {
// 			return 0, errors.New("is not a string or undefined")
// 		}
// 	} else {
// 		str = "latest"
// 	}

// 	switch str {
// 	case "latest":
// 		return -1, nil
// 	default:
// 		return int64(common.Big(v.(string)).Int64()), nil
// 	}
// }

// func hashString(raw interface{}, hash *string) error {
// 	argstr, ok := raw.(string)
// 	if !ok {
// 		return NewInvalidTypeError("", "not a string")
// 	}
// 	v := common.IsHex(argstr)
// 	hash = &argstr

// 	return nil
// }

type GetBlockByHashArgs struct {
	BlockHash  string
	IncludeTxs bool
}

func (args *GetBlockByHashArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}

	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 2 {
		return NewInsufficientParamsError(len(obj), 2)
	}

	argstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("blockHash", "not a string")
	}
	args.BlockHash = argstr

	args.IncludeTxs = obj[1].(bool)

	return nil
}

type GetBlockByNumberArgs struct {
	BlockNumber int64
	IncludeTxs  bool
}

func (args *GetBlockByNumberArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 2 {
		return NewInsufficientParamsError(len(obj), 2)
	}

	if v, ok := obj[0].(float64); ok {
		args.BlockNumber = int64(v)
	} else if v, ok := obj[0].(string); ok {
		args.BlockNumber = common.Big(v).Int64()
	} else {
		return NewInvalidTypeError("blockNumber", "not a number or string")
	}

	args.IncludeTxs = obj[1].(bool)

	return nil
}

type NewTxArgs struct {
	From     string
	To       string
	Value    *big.Int
	Gas      *big.Int
	GasPrice *big.Int
	Data     string

	BlockNumber int64
}

func (args *NewTxArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []json.RawMessage
	var ext struct {
		From     string
		To       string
		Value    interface{}
		Gas      interface{}
		GasPrice interface{}
		Data     string
	}

	// Decode byte slice to array of RawMessages
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	// Check for sufficient params
	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	// Decode 0th RawMessage to temporary struct
	if err := json.Unmarshal(obj[0], &ext); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(ext.From) == 0 {
		return NewValidationError("from", "is required")
	}

	args.From = ext.From
	args.To = ext.To
	args.Data = ext.Data

	var num int64
	if ext.Value == nil {
		return NewValidationError("value", "is required")
	} else {
		if err := numString(ext.Value, &num); err != nil {
			return err
		}
	}
	args.Value = big.NewInt(num)

	if ext.Gas == nil {
		return NewValidationError("gas", "is required")
	} else {
		if err := numString(ext.Gas, &num); err != nil {
			return err
		}
	}
	args.Gas = big.NewInt(num)

	if ext.GasPrice == nil {
		return NewValidationError("gasprice", "is required")
	} else {
		if err := numString(ext.GasPrice, &num); err != nil {
			return err
		}
	}
	args.GasPrice = big.NewInt(num)

	// Check for optional BlockNumber param
	if len(obj) > 1 {
		if err := blockHeightFromJson(obj[1], &args.BlockNumber); err != nil {
			return err
		}
	} else {
		args.BlockNumber = -1
	}

	return nil
}

type CallArgs struct {
	From     string
	To       string
	Value    *big.Int
	Gas      *big.Int
	GasPrice *big.Int
	Data     string

	BlockNumber int64
}

func (args *CallArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []json.RawMessage
	var ext struct {
		From     string
		To       string
		Value    interface{}
		Gas      interface{}
		GasPrice interface{}
		Data     string
	}

	// Decode byte slice to array of RawMessages
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	// Check for sufficient params
	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	// Decode 0th RawMessage to temporary struct
	if err := json.Unmarshal(obj[0], &ext); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(ext.From) == 0 {
		return NewValidationError("from", "is required")
	}
	args.From = ext.From

	if len(ext.To) == 0 {
		return NewValidationError("to", "is required")
	}
	args.To = ext.To

	var num int64
	if ext.Value == nil {
		num = int64(0)
	} else {
		if err := numString(ext.Value, &num); err != nil {
			return err
		}
	}
	args.Value = big.NewInt(num)

	if ext.Gas == nil {
		num = int64(0)
	} else {
		if err := numString(ext.Gas, &num); err != nil {
			return err
		}
	}
	args.Gas = big.NewInt(num)

	if ext.GasPrice == nil {
		num = int64(0)
	} else {
		if err := numString(ext.GasPrice, &num); err != nil {
			return err
		}
	}
	args.GasPrice = big.NewInt(num)

	args.Data = ext.Data

	// Check for optional BlockNumber param
	if len(obj) > 1 {
		if err := blockHeightFromJson(obj[1], &args.BlockNumber); err != nil {
			return err
		}
	} else {
		args.BlockNumber = -1
	}

	return nil
}

type GetStorageArgs struct {
	Address     string
	BlockNumber int64
}

func (args *GetStorageArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	addstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("address", "not a string")
	}
	args.Address = addstr

	if len(obj) > 1 {
		if err := blockHeight(obj[1], &args.BlockNumber); err != nil {
			return err
		}
	} else {
		args.BlockNumber = -1
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
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 2 {
		return NewInsufficientParamsError(len(obj), 2)
	}

	addstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("address", "not a string")
	}
	args.Address = addstr

	keystr, ok := obj[1].(string)
	if !ok {
		return NewInvalidTypeError("key", "not a string")
	}
	args.Key = keystr

	if len(obj) > 2 {
		if err := blockHeight(obj[2], &args.BlockNumber); err != nil {
			return err
		}
	} else {
		args.BlockNumber = -1
	}

	return nil
}

type GetTxCountArgs struct {
	Address     string
	BlockNumber int64
}

func (args *GetTxCountArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	addstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("address", "not a string")
	}
	args.Address = addstr

	if len(obj) > 1 {
		if err := blockHeight(obj[1], &args.BlockNumber); err != nil {
			return err
		}
	} else {
		args.BlockNumber = -1
	}

	return nil
}

type GetBalanceArgs struct {
	Address     string
	BlockNumber int64
}

func (args *GetBalanceArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	addstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("address", "not a string")
	}
	args.Address = addstr

	if len(obj) > 1 {
		if err := blockHeight(obj[1], &args.BlockNumber); err != nil {
			return err
		}
	} else {
		args.BlockNumber = -1
	}

	return nil
}

type GetDataArgs struct {
	Address     string
	BlockNumber int64
}

func (args *GetDataArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	addstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("address", "not a string")
	}
	args.Address = addstr

	if len(obj) > 1 {
		if err := blockHeight(obj[1], &args.BlockNumber); err != nil {
			return err
		}
	} else {
		args.BlockNumber = -1
	}

	return nil
}

type BlockNumArg struct {
	BlockNumber int64
}

func (args *BlockNumArg) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	if err := blockHeight(obj[0], &args.BlockNumber); err != nil {
		return err
	}

	return nil
}

type BlockNumIndexArgs struct {
	BlockNumber int64
	Index       int64
}

func (args *BlockNumIndexArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 2 {
		return NewInsufficientParamsError(len(obj), 2)
	}

	if err := blockHeight(obj[0], &args.BlockNumber); err != nil {
		return err
	}

	arg1, ok := obj[1].(string)
	if !ok {
		return NewInvalidTypeError("index", "not a string")
	}
	args.Index = common.Big(arg1).Int64()

	return nil
}

type HashArgs struct {
	Hash string
}

func (args *HashArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	arg0, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("hash", "not a string")
	}
	args.Hash = arg0

	return nil
}

type HashIndexArgs struct {
	Hash  string
	Index int64
}

func (args *HashIndexArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 2 {
		return NewInsufficientParamsError(len(obj), 2)
	}

	arg0, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("hash", "not a string")
	}
	args.Hash = arg0

	arg1, ok := obj[1].(string)
	if !ok {
		return NewInvalidTypeError("index", "not a string")
	}
	args.Index = common.Big(arg1).Int64()

	return nil
}

type Sha3Args struct {
	Data string
}

func (args *Sha3Args) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	argstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("data", "is not a string")
	}
	args.Data = argstr
	return nil
}

type BlockFilterArgs struct {
	Earliest int64
	Latest   int64
	Address  []string
	Topics   [][]string
	Skip     int
	Max      int
}

func (args *BlockFilterArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []struct {
		FromBlock interface{} `json:"fromBlock"`
		ToBlock   interface{} `json:"toBlock"`
		Limit     interface{} `json:"limit"`
		Offset    interface{} `json:"offset"`
		Address   interface{} `json:"address"`
		Topics    interface{} `json:"topics"`
	}

	if err = json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	// args.Earliest, err = toNumber(obj[0].ToBlock)
	// if err != nil {
	// 	return NewDecodeParamError(fmt.Sprintf("FromBlock %v", err))
	// }
	// args.Latest, err = toNumber(obj[0].FromBlock)
	// if err != nil {
	// 	return NewDecodeParamError(fmt.Sprintf("ToBlock %v", err))

	var num int64

	// if blank then latest
	if obj[0].FromBlock == nil {
		num = -1
	} else {
		if err := blockHeight(obj[0].FromBlock, &num); err != nil {
			return err
		}
	}
	// if -2 or other "silly" number, use latest
	if num < 0 {
		args.Earliest = -1 //latest block
	} else {
		args.Earliest = num
	}

	// if blank than latest
	if obj[0].ToBlock == nil {
		num = -1
	} else {
		if err := blockHeight(obj[0].ToBlock, &num); err != nil {
			return err
		}
	}
	args.Latest = num

	if obj[0].Limit == nil {
		num = defaultLogLimit
	} else {
		if err := numString(obj[0].Limit, &num); err != nil {
			return err
		}
	}
	args.Max = int(num)

	if obj[0].Offset == nil {
		num = defaultLogOffset
	} else {
		if err := numString(obj[0].Offset, &num); err != nil {
			return err
		}
	}
	args.Skip = int(num)

	if obj[0].Address != nil {
		marg, ok := obj[0].Address.([]interface{})
		if ok {
			v := make([]string, len(marg))
			for i, arg := range marg {
				argstr, ok := arg.(string)
				if !ok {
					return NewInvalidTypeError(fmt.Sprintf("address[%d]", i), "is not a string")
				}
				v[i] = argstr
			}
			args.Address = v
		} else {
			argstr, ok := obj[0].Address.(string)
			if ok {
				v := make([]string, 1)
				v[0] = argstr
				args.Address = v
			} else {
				return NewInvalidTypeError("address", "is not a string or array")
			}
		}
	}

	if obj[0].Topics != nil {
		other, ok := obj[0].Topics.([]interface{})
		if ok {
			topicdbl := make([][]string, len(other))
			for i, iv := range other {
				if argstr, ok := iv.(string); ok {
					// Found a string, push into first element of array
					topicsgl := make([]string, 1)
					topicsgl[0] = argstr
					topicdbl[i] = topicsgl
				} else if argarray, ok := iv.([]interface{}); ok {
					// Found an array of other
					topicdbl[i] = make([]string, len(argarray))
					for j, jv := range argarray {
						if v, ok := jv.(string); ok {
							topicdbl[i][j] = v
						} else {
							return NewInvalidTypeError(fmt.Sprintf("topic[%d][%d]", i, j), "is not a string")
						}
					}
				} else {
					return NewInvalidTypeError(fmt.Sprintf("topic[%d]", i), "not a string or array")
				}
			}
			args.Topics = topicdbl
			return nil
		} else {
			return NewInvalidTypeError("topic", "is not a string or array")
		}
	}

	return nil
}

type DbArgs struct {
	Database string
	Key      string
	Value    []byte
}

func (args *DbArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 2 {
		return NewInsufficientParamsError(len(obj), 2)
	}

	var objstr string
	var ok bool

	if objstr, ok = obj[0].(string); !ok {
		return NewInvalidTypeError("database", "not a string")
	}
	args.Database = objstr

	if objstr, ok = obj[1].(string); !ok {
		return NewInvalidTypeError("key", "not a string")
	}
	args.Key = objstr

	if len(obj) > 2 {
		objstr, ok = obj[2].(string)
		if !ok {
			return NewInvalidTypeError("value", "not a string")
		}

		args.Value = []byte(objstr)
	}

	return nil
}

func (a *DbArgs) requirements() error {
	if len(a.Database) == 0 {
		return NewValidationError("Database", "cannot be blank")
	}
	if len(a.Key) == 0 {
		return NewValidationError("Key", "cannot be blank")
	}
	return nil
}

type DbHexArgs struct {
	Database string
	Key      string
	Value    []byte
}

func (args *DbHexArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 2 {
		return NewInsufficientParamsError(len(obj), 2)
	}

	var objstr string
	var ok bool

	if objstr, ok = obj[0].(string); !ok {
		return NewInvalidTypeError("database", "not a string")
	}
	args.Database = objstr

	if objstr, ok = obj[1].(string); !ok {
		return NewInvalidTypeError("key", "not a string")
	}
	args.Key = objstr

	if len(obj) > 2 {
		objstr, ok = obj[2].(string)
		if !ok {
			return NewInvalidTypeError("value", "not a string")
		}

		args.Value = common.FromHex(objstr)
	}

	return nil
}

func (a *DbHexArgs) requirements() error {
	if len(a.Database) == 0 {
		return NewValidationError("Database", "cannot be blank")
	}
	if len(a.Key) == 0 {
		return NewValidationError("Key", "cannot be blank")
	}
	return nil
}

type WhisperMessageArgs struct {
	Payload  string
	To       string
	From     string
	Topics   []string
	Priority uint32
	Ttl      uint32
}

func (args *WhisperMessageArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []struct {
		Payload  string
		To       string
		From     string
		Topics   []string
		Priority interface{}
		Ttl      interface{}
	}

	if err = json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}
	args.Payload = obj[0].Payload
	args.To = obj[0].To
	args.From = obj[0].From
	args.Topics = obj[0].Topics

	var num int64
	if err := numString(obj[0].Priority, &num); err != nil {
		return err
	}
	args.Priority = uint32(num)

	if err := numString(obj[0].Ttl, &num); err != nil {
		return err
	}
	args.Ttl = uint32(num)

	return nil
}

type CompileArgs struct {
	Source string
}

func (args *CompileArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}
	argstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("arg0", "is not a string")
	}
	args.Source = argstr

	return nil
}

type FilterStringArgs struct {
	Word string
}

func (args *FilterStringArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	var argstr string
	argstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("filter", "not a string")
	}
	switch argstr {
	case "latest", "pending":
		break
	default:
		return NewValidationError("Word", "Must be `latest` or `pending`")
	}
	args.Word = argstr
	return nil
}

type FilterIdArgs struct {
	Id int
}

func (args *FilterIdArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	var num int64
	if err := numString(obj[0], &num); err != nil {
		return err
	}
	args.Id = int(num)

	return nil
}

type WhisperIdentityArgs struct {
	Identity string
}

func (args *WhisperIdentityArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err := json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	argstr, ok := obj[0].(string)
	if !ok {
		return NewInvalidTypeError("arg0", "not a string")
	}
	// if !common.IsHex(argstr) {
	// 	return NewValidationError("arg0", "not a hexstring")
	// }
	args.Identity = argstr

	return nil
}

type WhisperFilterArgs struct {
	To     string `json:"to"`
	From   string
	Topics []string
}

func (args *WhisperFilterArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []struct {
		To     interface{}
		Topics []interface{}
	}

	if err = json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 1 {
		return NewInsufficientParamsError(len(obj), 1)
	}

	var argstr string
	argstr, ok := obj[0].To.(string)
	if !ok {
		return NewInvalidTypeError("to", "is not a string")
	}
	args.To = argstr

	t := make([]string, len(obj[0].Topics))
	for i, j := range obj[0].Topics {
		argstr, ok := j.(string)
		if !ok {
			return NewInvalidTypeError("topics["+string(i)+"]", "is not a string")
		}
		t[i] = argstr
	}
	args.Topics = t

	return nil
}

type SubmitWorkArgs struct {
	Nonce  uint64
	Header string
	Digest string
}

func (args *SubmitWorkArgs) UnmarshalJSON(b []byte) (err error) {
	var obj []interface{}
	if err = json.Unmarshal(b, &obj); err != nil {
		return NewDecodeParamError(err.Error())
	}

	if len(obj) < 3 {
		return NewInsufficientParamsError(len(obj), 3)
	}

	var objstr string
	var ok bool
	if objstr, ok = obj[0].(string); !ok {
		return NewInvalidTypeError("nonce", "not a string")
	}

	args.Nonce = common.String2Big(objstr).Uint64()
	if objstr, ok = obj[1].(string); !ok {
		return NewInvalidTypeError("header", "not a string")
	}

	args.Header = objstr

	if objstr, ok = obj[2].(string); !ok {
		return NewInvalidTypeError("digest", "not a string")
	}

	args.Digest = objstr

	return nil
}
