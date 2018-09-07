package cuckoo

import (
	"math/big"

	"github.com/ethereum/go-ethereum/common"
)

type Block interface {
	Difficulty() *big.Int
//	HashNoNonce() common.Hash
	Nonce() uint64
	MixDigest() common.Hash
	NumberU64() uint64
}
