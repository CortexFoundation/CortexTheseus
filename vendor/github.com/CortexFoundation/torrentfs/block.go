package torrentfs

import (
	"crypto/sha256"
	"github.com/CortexFoundation/torrentfs/merkletree"
)

type BlockContent struct {
	x string
	n uint64
}

func (t BlockContent) CalculateHash() ([]byte, error) {
	h := sha256.New()
	if _, err := h.Write([]byte(t.x)); err != nil {
		return nil, err
	}

	return h.Sum(nil), nil
}

//Equals tests for equality of two Contents
func (t BlockContent) Equals(other merkletree.Content) (bool, error) {
	return t.x == other.(BlockContent).x, nil
}
