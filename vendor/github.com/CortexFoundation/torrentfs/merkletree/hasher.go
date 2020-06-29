package merkletree

import (
	"hash"
	"sync"

	"github.com/CortexFoundation/CortexTheseus/metrics"
	"golang.org/x/crypto/sha3"
)

var (
	hashSumMeter = metrics.NewRegisteredMeter("torrent/hash/summary", nil)
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
	hashSumMeter.Mark(1)
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
