package torrentfs

import (
	"github.com/CortexFoundation/torrentfs/merkletree"
	"golang.org/x/crypto/sha3"
)

type BlockContent struct {
	x string
	n uint64
}

func (t BlockContent) CalculateHash() ([]byte, error) {
	h := sha3.NewLegacyKeccak256()
	if _, err := h.Write([]byte(t.x)); err != nil {
		return nil, err
	}

	return h.Sum(nil), nil
}

//Equals tests for equality of two Contents
func (t BlockContent) Equals(other merkletree.Content) (bool, error) {
	return t.x == other.(BlockContent).x, nil
}
