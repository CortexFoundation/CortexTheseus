// Copyright 2020 The CortexTheseus Authors
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
package types

import (
	"bytes"
	//"errors"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/rlp"
	"github.com/anacrolix/torrent/metainfo"
	"github.com/CortexFoundation/torrentfs/params"
)

const (
	opCommon      = 0
	opCreateModel = 1
	opCreateInput = 2
	opNoInput     = 3
)

type FileInfo struct {
	Meta *FileMeta
	// Transaction hash
	//TxHash *common.Hash
	// Contract Address
	ContractAddr *common.Address
	LeftSize     uint64
	Relate       []common.Address
}

//var (
//	errWrongOpCode = errors.New("unexpected opCode")
//)

// Transaction ... Tx struct
type Transaction struct {
	//Price     *big.Int        `json:"gasPrice" gencodec:"required"`
	Amount   *big.Int `json:"value"    gencodec:"required"`
	GasLimit uint64   `json:"gas"      gencodec:"required"`
	Payload  []byte   `json:"input"    gencodec:"required"`
	//From      *common.Address `json:"from"     gencodec:"required"`
	Recipient *common.Address `json:"to"       rlp:"nil"` // nil means contract creation
	Hash      *common.Hash    `json:"hash"     gencodec:"required"`
	//Receipt   *TxReceipt      `json:"receipt"  rlp:"nil"`
}

// Op ...
func (t *Transaction) Op() (op int) {
	op = opCommon
	if len(t.Payload) >= 2 {
		op = (int(t.Payload[0]) << 8) + int(t.Payload[1])
		if op > 3 {
			op = opNoInput
		}
	} else if len(t.Payload) == 0 {
		op = opNoInput
	}
	return
}

// Data ...
func (t *Transaction) Data() []byte {
	if len(t.Payload) >= 2 {
		return t.Payload[2:]
	}
	return []byte{}
}

//func (t *Transaction) noPayload() bool {
//	return len(t.Payload) == 0
//}

// IsFlowControl ...
func (t *Transaction) IsFlowControl() bool {
	return t.Amount.Sign() == 0 && t.GasLimit >= params.UploadGas
}

func (t *Transaction) Parse() *FileMeta {
	if t.Op() == opCreateInput {
		var meta types.InputMeta
		if err := rlp.Decode(bytes.NewReader(t.Data()), &meta); err != nil {
			return nil
		}
		var InfoHash = meta.InfoHash()
		return &FileMeta{
			InfoHash,
			//	meta.Comment,
			meta.RawSize,
			//meta.BlockNum.Uint64(),
		}
	} else if t.Op() == opCreateModel {
		var meta types.ModelMeta
		if err := rlp.Decode(bytes.NewReader(t.Data()), &meta); err != nil {
			return nil
		}
		var InfoHash = meta.InfoHash()
		return &FileMeta{
			InfoHash,
			//	meta.Comment,
			meta.RawSize,
			//meta.BlockNum.Uint64(),
		}
	} else {
		return nil
	}
}

//type transactionMarshaling struct {
//Price    *hexutil.Big
//	Amount   *hexutil.Big
//	GasLimit hexutil.Uint64
//	Payload  hexutil.Bytes
//}

// gencodec -type Block -field-override blockMarshaling -out gen_block_json.go
// Block ... block struct
type Block struct {
	Number uint64      `json:"number"           gencodec:"required"`
	Hash   common.Hash `json:"Hash"             gencodec:"required"`
	//ParentHash common.Hash   `json:"parentHash"       gencodec:"required"`
	Txs []Transaction `json:"Transactions"     gencodec:"required"`
}

type blockMarshaling struct {
	Number hexutil.Uint64
}

// TxReceipt ...
type TxReceipt struct {
	// Contract Address
	ContractAddr *common.Address `json:"ContractAddress"  gencodec:"required"`
	// Transaction Hash
	TxHash *common.Hash `json:"TransactionHash"  gencodec:"required"`
	//Receipt   *TxReceipt      `json:"receipt"  rlp:"nil"`
	GasUsed uint64 `json:"gasUsed" gencodec:"required"`
	Status  uint64 `json:"status"`
}

// FileMeta ...
type FileMeta struct {
	InfoHash metainfo.Hash `json:"InfoHash"         gencodec:"required"`
	//	Name     string        `json:"Name"             gencodec:"required"`
	// The raw size of the file counted in bytes
	RawSize uint64 `json:"RawSize"          gencodec:"required"`
	//BlockNum uint64 `json:"BlockNum"         gencodec:"required"`
}

// DisplayName ...
//func (m *FileMeta) DisplayName() string {
//	return m.Name
//}

type FlowControlMeta struct {
	InfoHash       metainfo.Hash
	BytesRequested uint64
	IsCreate       bool
}
