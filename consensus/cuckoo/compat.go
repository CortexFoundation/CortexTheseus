package cuckoo

import (
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
)

type Block interface {
	Difficulty() *big.Int
	//	HashNoNonce() common.Hash
	Nonce() uint64
	MixDigest() common.Hash
	NumberU64() uint64
}
