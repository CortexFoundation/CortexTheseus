package main

/*
#cgo  LDFLAGS:  -lstdc++  -lgominer
#include "gominer.h"
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"math/big"
	"unsafe"
)

type Hash [32]byte
type Address [20]byte
type Bloom [256]byte

type BlockNonce [8]byte
type BlockSolution [42]uint32

type Header struct {
	ParentHash  Hash          `json:"parentHash"       gencodec:"required"`
	UncleHash   Hash          `json:"sha3Uncles"       gencodec:"required"`
	Coinbase    Address       `json:"miner"            gencodec:"required"`
	Root        Hash          `json:"stateRoot"        gencodec:"required"`
	TxHash      Hash          `json:"transactionsRoot" gencodec:"required"`
	ReceiptHash Hash          `json:"receiptsRoot"     gencodec:"required"`
	Bloom       Bloom         `json:"logsBloom"        gencodec:"required"`
	Difficulty  *big.Int      `json:"difficulty"       gencodec:"required"`
	Number      *big.Int      `json:"number"           gencodec:"required"`
	GasLimit    uint64        `json:"gasLimit"         gencodec:"required"`
	GasUsed     uint64        `json:"gasUsed"          gencodec:"required"`
	Time        *big.Int      `json:"timestamp"        gencodec:"required"`
	Extra       []byte        `json:"extraData"        gencodec:"required"`
	MixDigest   Hash          `json:"mixHash"          gencodec:"required"`
	Nonce       BlockNonce    `json:"nonce"            gencodec:"required"`
	Solution    BlockSolution `json:"solution"			gencodec:"required"`
}

func main() {
	C.CuckooInit()
	header := &Header{
		Number:     big.NewInt(1),
		Difficulty: big.NewInt(100),
		Time:       big.NewInt(0),
	}

	for nonce := 0; ; nonce++ {
		fmt.Println("Trying solve with nonce ", nonce)

		var result BlockSolution
		var result_len uint32

		out, _ := json.Marshal(header)
		var header_len = unsafe.Sizeof(out)
		C.CuckooSolve(
			(*C.char)(unsafe.Pointer(&out[0])),
			C.uint(header_len),
			C.uint(nonce),
			(*C.uint)(unsafe.Pointer(&result[0])),
			(*C.uint)(unsafe.Pointer(&result_len)))

		r := C.CuckooVerify(
			(*C.char)(unsafe.Pointer(&out[0])),
			C.uint(header_len),
			C.uint(nonce),
			(*C.uint)(unsafe.Pointer(&result[0])))

		fmt.Println(header)

		if byte(r) != 0 {
			// binary.BigEndian.PutUint64(header.nonce[:], 63)
			// header.result = result
			fmt.Println(nonce, result)
			break
		}
	}

}
