// Copyright 2019 The go-ethereum Authors
// This file is part of The go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package trie

import (
	"sync"

	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

// hasher is a type used for the trie Hash operation. A hasher has some
// internal preallocated temp space
type hasher struct {
	sha      crypto.KeccakState
	tmp      []byte
	encbuf   rlp.EncoderBuffer
	parallel bool // Whether to use paralallel threads when hashing
}

// hasherPool holds pureHashers
var hasherPool = sync.Pool{
	New: func() any {
		return &hasher{
			tmp:    make([]byte, 0, 550), // cap is as large as a full fullNode.
			sha:    crypto.NewKeccakState(),
			encbuf: rlp.NewEncoderBuffer(nil),
		}
	},
}

func newHasher(parallel bool) *hasher {
	h := hasherPool.Get().(*hasher)
	h.parallel = parallel
	return h
}

func returnHasherToPool(h *hasher) {
	hasherPool.Put(h)
}

// hash collapses a node down into a hash node, also returning a copy of the
// original node initialized with the computed hash to replace the original one.
func (h *hasher) hash(n node, force bool) node {
	// Return the cached hash if it's available
	if hash, _ := n.cache(); hash != nil {
		return hash
	}
	// Trie not processed yet, walk the children
	switch n := n.(type) {
	case *shortNode:
		collapsed := h.hashShortNodeChildren(n)
		hashed := h.shortnodeToHash(collapsed, force)
		// We need to retain the possibly _not_ hashed node, in case it was too
		// small to be hashed
		if hn, ok := hashed.(hashNode); ok {
			n.flags.hash = hn
		} else {
			n.flags.hash = nil
		}
		return hashed
	case *fullNode:
		collapsed := h.hashFullNodeChildren(n)
		hashed := h.fullnodeToHash(collapsed, force)
		if hn, ok := hashed.(hashNode); ok {
			n.flags.hash = hn
		} else {
			n.flags.hash = nil
		}
		return hashed
	default:
		// Value and hash nodes don't have children, so they're left as were
		return n
	}
}

// hashShortNodeChildren collapses the short node. The returned collapsed node
// holds a live reference to the Key, and must not be modified.
func (h *hasher) hashShortNodeChildren(n *shortNode) *shortNode {
	// Hash the short node's child, caching the newly hashed subtree
	//collapsed, cached = n.copy(), n.copy()
	var collapsed shortNode
	// Previously, we did copy this one. We don't seem to need to actually
	// do that, since we don't overwrite/reuse keys
	// cached.Key = common.CopyBytes(n.Key)
	collapsed.Key = hexToCompact(n.Key)
	// Unless the child is a valuenode or hashnode, hash it
	switch n.Val.(type) {
	case *fullNode, *shortNode:
		collapsed.Val = h.hash(n.Val, false)
	default:
		collapsed.Val = n.Val
	}
	return &collapsed
}

func (h *hasher) hashFullNodeChildren(n *fullNode) *fullNode {
	// Hash the full node's children, caching the newly hashed subtrees
	var children [17]node
	if h.parallel {
		var wg sync.WaitGroup
		for i := 0; i < 16; i++ {
			if child := n.Children[i]; child != nil {
				wg.Add(1)
				go func(i int) {
					hasher := newHasher(false)
					children[i] = hasher.hash(child, false)
					returnHasherToPool(hasher)
					wg.Done()
				}(i)
			} else {
				children[i] = nilValueNode
			}
		}
		wg.Wait()
	} else {
		for i := 0; i < 16; i++ {
			if child := n.Children[i]; child != nil {
				children[i] = h.hash(child, false)
			} else {
				children[i] = nilValueNode
			}
		}
	}
	if n.Children[16] != nil {
		children[16] = n.Children[16]
	}
	return &fullNode{flags: nodeFlag{}, Children: children}
}

// shortnodeToHash creates a hashNode from a shortNode. The supplied shortnode
// should have hex-type Key, which will be converted (without modification)
// into compact form for RLP encoding.
// If the rlp data is smaller than 32 bytes, `nil` is returned.
func (h *hasher) shortnodeToHash(n *shortNode, force bool) node {
	n.encode(h.encbuf)
	enc := h.encodedBytes()

	if len(enc) < 32 && !force {
		return n // Nodes smaller than 32 bytes are stored inside their parent
	}
	return h.hashData(enc)
}

// fullnodeToHash is used to create a hashNode from a fullNode, (which
// may contain nil values)
func (h *hasher) fullnodeToHash(n *fullNode, force bool) node {
	n.encode(h.encbuf)
	enc := h.encodedBytes()

	if len(enc) < 32 && !force {
		return n // Nodes smaller than 32 bytes are stored inside their parent
	}
	return h.hashData(enc)
}

// encodedBytes returns the result of the last encoding operation on h.encbuf.
// This also resets the encoder buffer.
//
// All node encoding must be done like this:
//
//	node.encode(h.encbuf)
//	enc := h.encodedBytes()
//
// This convention exists because node.encode can only be inlined/escape-analyzed when
// called on a concrete receiver type.
func (h *hasher) encodedBytes() []byte {
	h.tmp = h.encbuf.AppendToBytes(h.tmp[:0])
	h.encbuf.Reset(nil)
	return h.tmp
}

// hashData hashes the provided data
func (h *hasher) hashData(data []byte) hashNode {
	n := make(hashNode, 32)
	h.sha.Reset()
	h.sha.Write(data)
	h.sha.Read(n)
	return n
}

// hashDataTo hashes the provided data to the given destination buffer. The caller
// must ensure that the dst buffer is of appropriate size.
func (h *hasher) hashDataTo(dst, data []byte) {
	h.sha.Reset()
	h.sha.Write(data)
	h.sha.Read(dst)
}

// proofHash is used to construct trie proofs, and returns the 'collapsed'
// node (for later RLP encoding) aswell as the hashed node -- unless the
// node is smaller than 32 bytes, in which case it will be returned as is.
// This method does not do anything on value- or hash-nodes.
func (h *hasher) proofHash(original node) (collapsed, hashed node) {
	switch n := original.(type) {
	case *shortNode:
		sn := h.hashShortNodeChildren(n)
		return sn, h.shortnodeToHash(sn, false)
	case *fullNode:
		fn := h.hashFullNodeChildren(n)
		return fn, h.fullnodeToHash(fn, false)
	default:
		// Value and hash nodes don't have children, so they're left as were
		return n, n
	}
}
