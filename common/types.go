package common

import (
	"encoding/json"
	"errors"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"math/big"
)

// Transaction ... Tx struct
type Transaction struct {
	Price        *big.Int        `json:"gasPrice" gencodec:"required"`
	GasLimit     uint64          `json:"gas"      gencodec:"required"`
	Recipient    *common.Address `json:"to"       rlp:"nil"` // nil means contract creation
	Amount       *big.Int        `json:"value"    gencodec:"required"`
	Payload      []byte          `json:"input"    gencodec:"required"`
	From         *common.Address `json:"to"       gencodec:"required"`
	Hash         *common.Hash    `json:"hash"     rlp:"-"`
}

func (t *Transaction) UnmarshalJSON(input []byte) error {
	type Transaction struct {
		Price        *hexutil.Big    `json:"gasPrice" gencodec:"required"`
		GasLimit     *hexutil.Uint64 `json:"gas"      gencodec:"required"`
		Recipient    *common.Address `json:"to"       rlp:"nil"`
		From         *common.Address `json:"from"     gencodec:"required"`
		Amount       *hexutil.Big    `json:"value"    gencodec:"required"`
		Payload      *hexutil.Bytes  `json:"input"    gencodec:"required"`
		Hash         *common.Hash    `json:"hash"     rlp:"-"`
	}
	var dec Transaction
	if err := json.Unmarshal(input, &dec); err != nil {
		return err
	}
	if dec.Price == nil {
		return errors.New("missing required field 'gasPrice' for txdata")
	}
	t.Price = (*big.Int)(dec.Price)
	if dec.GasLimit == nil {
		return errors.New("missing required field 'gas' for txdata")
	}
	t.GasLimit = uint64(*dec.GasLimit)
	if dec.Recipient != nil {
		t.Recipient = dec.Recipient
	}
	if dec.From != nil {
		t.From = dec.From
	}
	if dec.Amount == nil {
		return errors.New("missing required field 'value' for txdata")
	}
	t.Amount = (*big.Int)(dec.Amount)
	if dec.Payload == nil {
		return errors.New("missing required field 'input' for txdata")
	}
	t.Payload = *dec.Payload
	if dec.Hash != nil {
		t.Hash = dec.Hash
	}
	return nil
}


// Block ... block struct
type Block struct {
	Number     uint64              `json:"number"           gencodec:"required"`
	Hash       common.Hash         `json:"Hash"             gencodec:"required"`
	ParentHash common.Hash         `json:"parentHash"       gencodec:"required"`
	Txs        []Transaction       `json:"Transactions"     gencodec:"required"`
}

func (h *Block) UnmarshalJSON(input []byte) error {
	type Block struct {
		Number     *hexutil.Uint64      `json:"number"           gencodec:"required"`
		Hash       *common.Hash         `json:"Hash"             gencodec:"required"`
		ParentHash *common.Hash         `json:"parentHash"       gencodec:"required"`
		Txs        *[]Transaction       `json:"Transactions"     gencodec:"required"`
	}
	var dec Block
	if err := json.Unmarshal(input, &dec); err != nil {
		return err
	}
	if dec.ParentHash == nil {
		return errors.New("missing required field 'parentHash' for Block")
	}
	h.ParentHash = *dec.ParentHash
	if dec.Hash == nil {
		return errors.New("missing required field 'Hash' for Block")
	}
	h.Hash = *dec.Hash
	if dec.Number == nil {
		return errors.New("missing required field 'Number' for Block")
	}
	h.Number = uint64(*dec.Number)
	h.Txs = *dec.Txs
	return nil
}


// Receipt ...
type Receipt struct {
	// Contract Address
	ContractAddr string `json:"ContractAddr"`
	// Transaction Hash
	TxHash       string `json:"TransactionHash"`
}

// FileMeta ...
type FileMeta struct {
	// Transaction hash
	TxHash       string
	// Contract Address
	ContractAddr string
	// Author Address
	AuthorAddr   string
	// Download URI, should be in magnetURI format
	URI          string
	// The raw size of the file counted in bytes
	RawSize      uint64
	BlockNum     uint64
}

// File
type FileInfo struct {
	FileMeta
}

// FileStorage ...
type FileStorage struct {
	files []*FileMeta
}

// FlowControlMeta ...
type FlowControlMeta struct {
	URI            string
	BytesRequested uint64
}
