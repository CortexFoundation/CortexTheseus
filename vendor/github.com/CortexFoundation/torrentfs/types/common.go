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
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/rlp"

	"github.com/CortexFoundation/torrentfs/params"
)

const (
	opCommon      = 0
	opCreateModel = 1
	opCreateInput = 2
	opNoInput     = 3
)

//go:generate go run github.com/fjl/gencodec -type FileInfo -out gen_fileinfo_json.go
//go:generate go run github.com/CortexFoundation/CortexTheseus/rlp/rlpgen -type FileInfo -out gen_fileinfo_rlp.go

type FileInfo struct {
	Meta *FileMeta
	// Transaction hash
	//TxHash *common.Hash
	// Contract Address
	ContractAddr *common.Address
	LeftSize     uint64
	Relate       []common.Address
}

//go:generate go run github.com/fjl/gencodec -type Transaction -field-override transactionMarshaling -out gen_tx_json.go
//go:generate go run github.com/CortexFoundation/CortexTheseus/rlp/rlpgen -type Transaction -out gen_tx_rlp.go

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
		var meta InputMeta
		if err := rlp.DecodeBytes(t.Data(), &meta); err != nil {
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
		var meta ModelMeta
		if err := rlp.DecodeBytes(t.Data(), &meta); err != nil {
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

type transactionMarshaling struct {
	Amount   *hexutil.Big
	GasLimit hexutil.Uint64
	Payload  hexutil.Bytes
}

//go:generate go run github.com/fjl/gencodec -type Block -field-override blockMarshaling -out gen_block_json.go
//go:generate go run github.com/CortexFoundation/CortexTheseus/rlp/rlpgen -type Block -out gen_block_rlp.go

type Block struct {
	Number uint64      `json:"number"           gencodec:"required"`
	Hash   common.Hash `json:"Hash"             gencodec:"required"`
	//ParentHash common.Hash   `json:"parentHash"       gencodec:"required"`
	Txs []Transaction `json:"transactions"     gencodec:"required"`
}

type blockMarshaling struct {
	Number hexutil.Uint64
}

//go:generate go run github.com/fjl/gencodec -type Receipt -field-override receiptMarshaling -out gen_receipt_json.go
//go:generate go run github.com/CortexFoundation/CortexTheseus/rlp/rlpgen -type Receipt -out gen_receipt_rlp.go
type Receipt struct {
	// Contract Address
	ContractAddr *common.Address `json:"contractAddress"`
	// Transaction Hash
	//TxHash *common.Hash `json:"transactionHash"  gencodec:"required"`
	//Receipt   *TxReceipt      `json:"receipt"  rlp:"nil"`
	GasUsed uint64 `json:"gasUsed" gencodec:"required"`
	Status  uint64 `json:"status"`
}

type receiptMarshaling struct {
	Status  hexutil.Uint64
	GasUsed hexutil.Uint64
}

//go:generate go run github.com/fjl/gencodec -type FileMeta -out gen_filemeta_json.go
//go:generate go run github.com/CortexFoundation/CortexTheseus/rlp/rlpgen -type FileMeta -out gen_filemeta_rlp.go
type FileMeta struct {
	InfoHash string `json:"infoHash"         gencodec:"required"`
	//	Name     string        `json:"Name"             gencodec:"required"`
	// The raw size of the file counted in bytes
	RawSize uint64 `json:"rawSize"          gencodec:"required"`
	//BlockNum uint64 `json:"BlockNum"         gencodec:"required"`
}

func (fm *FileMeta) DecodeRLP(s *rlp.Stream) error {
	var dec FileMeta
	if err := s.Decode(&dec); err != nil {
		return err
	}
	fm.InfoHash, fm.RawSize = dec.InfoHash, dec.RawSize
	return nil
}

// DisplayName ...
//func (m *FileMeta) DisplayName() string {
//	return m.Name
//}

type BitsFlow struct {
	infohash string
	request  uint64
	//IsCreate       bool
}

func (b *BitsFlow) InfoHash() string {
	return b.infohash
}

func (b *BitsFlow) Request() uint64 {
	return b.request
}

func NewBitsFlow(ih string, request uint64) *BitsFlow {
	return &BitsFlow{
		infohash: ih,
		request:  request,
	}
}
