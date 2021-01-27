package merkletree

import (
	"hash"
	"sync"

	"golang.org/x/crypto/sha3"
)

type keccakState interface {
	hash.Hash
	Read([]byte) (int, error)
}

type hashNode []byte

type hasher struct {
	sha keccakState
}

var hasherPool = sync.Pool{
	New: func() interface{} {
		return &hasher{
			sha: sha3.NewLegacyKeccak256().(keccakState),
		}
	},
}

func (h *hasher) sum(data []byte) []byte {
	n := make([]byte, 32)
	h.sha.Reset()
	h.sha.Write(data)
	h.sha.Read(n)
	return n
}

func newHasher() *hasher {
	h := hasherPool.Get().(*hasher)
	return h
}

func returnHasherToPool(h *hasher) {
	hasherPool.Put(h)
}
