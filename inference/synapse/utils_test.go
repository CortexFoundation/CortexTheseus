package synapse

import (
	"testing"
)

func TestRLPHashString(t *testing.T) {
	var rlp string
	a := []byte{}
	rlp = RLPHashString(a)
	t.Log("rlp hash", "data", a, "rlp", rlp)
}
