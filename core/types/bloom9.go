package types

import (
	"math/big"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/state"
)

func CreateBloom(receipts Receipts) []byte {
	bin := new(big.Int)
	for _, receipt := range receipts {
		bin.Or(bin, LogsBloom(receipt.logs))
	}

	return ethutil.LeftPadBytes(bin.Bytes(), 256)
}

func LogsBloom(logs state.Logs) *big.Int {
	bin := new(big.Int)
	for _, log := range logs {
		data := make([][]byte, len(log.Topics())+1)
		data[0] = log.Address()

		for i, topic := range log.Topics() {
			data[i+1] = topic
		}

		for _, b := range data {
			bin.Or(bin, ethutil.BigD(bloom9(crypto.Sha3(b)).Bytes()))
		}
	}

	return bin
}

func bloom9(b []byte) *big.Int {
	r := new(big.Int)

	for i := 0; i < 16; i += 2 {
		t := big.NewInt(1)
		b := uint(b[i+1]) + 1024*(uint(b[i])&1)
		r.Or(r, t.Lsh(t, b))
	}

	return r
}

func BloomLookup(bin, topic []byte) bool {
	bloom := ethutil.BigD(bin)
	cmp := bloom9(crypto.Sha3(topic))

	return bloom.And(bloom, cmp).Cmp(cmp) == 0
}
