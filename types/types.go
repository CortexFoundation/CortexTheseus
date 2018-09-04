package types

import (
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
)

// Transaction ... Tx struct
type Transaction struct {
	Price     *big.Int        `json:"gasPrice" gencodec:"required"`
	Amount    *big.Int        `json:"value"    gencodec:"required"`
	GasLimit  uint64          `json:"gas"      gencodec:"required"`
	Payload   []byte          `json:"input"    gencodec:"required"`
	From      *common.Address `json:"from"     gencodec:"required"`
	Recipient *common.Address `json:"to"       rlp:"nil"` // nil means contract creation
	Hash      *common.Hash    `json:"hash"     rlp:"-"`
}

type transactionMarshaling struct {
	Price    *hexutil.Big
	Amount   *hexutil.Big
	GasLimit hexutil.Uint64
	Payload  hexutil.Bytes
}

// Block ... block struct
type Block struct {
	Number     uint64        `json:"number"           gencodec:"required"`
	Hash       common.Hash   `json:"Hash"             gencodec:"required"`
	ParentHash common.Hash   `json:"parentHash"       gencodec:"required"`
	Txs        []Transaction `json:"Transactions"     gencodec:"required"`
}

type blockMarshaling struct {
	Number hexutil.Uint64
}

// Receipt ...
type Receipt struct {
	// Contract Address
	ContractAddr *common.Address `json:"ContractAddr"`
	// Transaction Hash
	TxHash *common.Hash `json:"TransactionHash"`
}

// FileMeta ...
type FileMeta struct {
	// Transaction hash
	TxHash *common.Hash
	// Contract Address
	ContractAddr *common.Address
	// Author Address
	AuthorAddr *common.Address
	// Download URI, should be in magnetURI format
	URI string
	// The raw size of the file counted in bytes
	RawSize  uint64
	BlockNum uint64
}

// FileInfo ...
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
