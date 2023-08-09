// Copyright 2021 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

// Package core implements the Cortex consensus protocol

package merkletree

import (
	"bytes"
	"errors"
	"fmt"
	//"golang.org/x/crypto/sha3"
	//"hash"
)

// Content represents the data that is stored and verified by the tree. A type that
// implements this interface can be used as an item in the tree.
type Content interface {
	CalculateHash() ([]byte, error)
	Equals(other Content) (bool, error)
}

// MerkleTree is the container for the tree. It holds a pointer to the root of the tree,
// a list of pointers to the leaf nodes, and the merkle root.
type MerkleTree struct {
	Root       *Node
	merkleRoot []byte
	Leafs      []*Node
	//hashStrategy func() hash.Hash
}

// Node represents a node, root, or leaf in the tree. It stores pointers to its immediate
// relationships, a hash, the content stored if it is a leaf, and other metadata.
type Node struct {
	Tree   *MerkleTree
	Parent *Node
	Left   *Node
	Right  *Node
	leaf   bool
	dup    bool
	Hash   []byte
	C      Content
}

// verifyNode walks down the tree until hitting a leaf, calculating the hash at each level
// and returning the resulting hash of Node n.
func (n *Node) verifyNode() ([]byte, error) {
	if n.leaf {
		return n.C.CalculateHash()
	}
	rightBytes, err := n.Right.verifyNode()
	if err != nil {
		return nil, err
	}

	leftBytes, err := n.Left.verifyNode()
	if err != nil {
		return nil, err
	}

	//h := n.Tree.hashStrategy()
	//if _, err := h.Write(append(leftBytes, rightBytes...)); err != nil {
	//	return nil, err
	//}
	//return h.Sum(nil), nil
	return n.Tree.safeHash(append(leftBytes, rightBytes...)), nil
}

// calculateNodeHash is a helper function that calculates the hash of the node.
func (n *Node) calculateNodeHash() ([]byte, error) {
	if n.leaf {
		return n.C.CalculateHash()
	}

	//h := n.Tree.hashStrategy()
	//if _, err := h.Write(append(n.Left.Hash, n.Right.Hash...)); err != nil {
	//	return nil, err
	//}

	//return h.Sum(nil), nil
	return n.Tree.safeHash(append(n.Left.Hash, n.Right.Hash...)), nil
}

// NewTree creates a new Merkle Tree using the content cs.
func NewTree(cs []Content) (*MerkleTree, error) {
	//var defaultHashStrategy = sha3.NewLegacyKeccak256
	t := &MerkleTree{
		//hashStrategy: defaultHashStrategy,
	}
	root, leafs, err := buildWithContent(cs, t)
	if err != nil {
		return nil, err
	}
	t.Root = root
	t.Leafs = leafs
	t.merkleRoot = root.Hash
	return t, nil
}

//NewTreeWithHashStrategy creates a new Merkle Tree using the content cs using the provided hash
//strategy. Note that the hash type used in the type that implements the Content interface must
//match the hash type profided to the tree.
//func NewTreeWithHashStrategy(cs []Content, hashStrategy func() hash.Hash) (*MerkleTree, error) {
//	t := &MerkleTree{
//		hashStrategy: hashStrategy,
//	}
//	root, leafs, err := buildWithContent(cs, t)
//	if err != nil {
//		return nil, err
//	}
//	t.Root = root
//	t.Leafs = leafs
//	t.merkleRoot = root.Hash
//	return t, nil
//}

func (m *MerkleTree) safeHash(data []byte) []byte {
	h := newHasher()
	defer returnHasherToPool(h)
	return h.sum(data)
}

func (m *MerkleTree) AddNode(c Content) error {
	hash, err := c.CalculateHash()
	if err != nil {
		return err
	}
	n := len(m.Leafs)
	// If n is 0, which means the MerkleTree is empty
	// The new leaf should be the root of the MerkleTree
	if n == 0 {
		newLeaf := &Node{
			Hash: hash,
			C:    c,
			leaf: true,
			dup:  false,
			Tree: m,
		}
		dupLeaf := &Node{
			Hash: hash,
			C:    c,
			leaf: true,
			dup:  true,
			Tree: m,
		}
		//h := m.hashStrategy()
		//if _, err := h.Write(append(newLeaf.Hash, dupLeaf.Hash...)); err != nil {
		//	return err
		//}
		root := &Node{
			Tree:  m,
			Left:  newLeaf,
			Right: dupLeaf,
			//Hash:  h.Sum(nil),
			Hash: m.safeHash(append(newLeaf.Hash, dupLeaf.Hash...)),
			C:    nil,
		}
		newLeaf.Parent = root
		dupLeaf.Parent = root
		m.Root = root
		m.merkleRoot = root.Hash
		return nil
	}
	// If the last leaf is a duplicated node, the new leaf can replace it and update the hash of influenced nodes.
	// Otherwise, the new leaf is replicated and a new path is created to the root.
	if m.Leafs[n-1].dup {
		newLeaf := &Node{
			Hash:   hash,
			C:      c,
			leaf:   true,
			dup:    false,
			Tree:   m,
			Parent: m.Leafs[n-1].Parent,
		}
		newLeaf.Parent.Right = newLeaf
		m.Leafs[n-1] = newLeaf
		for ; newLeaf.Parent != nil; newLeaf = newLeaf.Parent {
			//h := m.hashStrategy()
			//if _, err := h.Write(append(newLeaf.Parent.Left.Hash, newLeaf.Hash...)); err != nil {
			//	return err
			//}
			newLeaf.Parent.Hash = m.safeHash(append(newLeaf.Parent.Left.Hash, newLeaf.Hash...)) //h.Sum(nil)
		}
	} else {
		newLeaf := &Node{
			Hash: hash,
			C:    c,
			leaf: true,
			dup:  false,
			Tree: m,
		}
		dupLeaf := &Node{
			Hash: hash,
			C:    c,
			leaf: true,
			dup:  true,
			Tree: m,
		}
		m.Leafs = append(m.Leafs, newLeaf)
		m.Leafs = append(m.Leafs, dupLeaf)
		// First, the new path is created if the number of original nodes in this layer is even.
		//h := m.hashStrategy()
		//if _, err := h.Write(append(newLeaf.Hash, dupLeaf.Hash...)); err != nil {
		//	return err
		//}
		node := &Node{
			Tree:  m,
			Left:  newLeaf,
			Right: dupLeaf,
			//Hash:  h.Sum(nil),
			Hash: m.safeHash(append(newLeaf.Hash, dupLeaf.Hash...)),
		}
		newLeaf.Parent = node
		dupLeaf.Parent = node
		lastNode := m.Leafs[n-1].Parent
		for n /= 2; n%2 == 0; n /= 2 {
			//h = m.hashStrategy()
			//if _, err := h.Write(append(node.Hash, node.Hash...)); err != nil {
			//	return err
			//}
			parentNode := &Node{
				Tree:  m,
				Left:  node,
				Right: node,
				Hash:  m.safeHash(append(node.Hash, node.Hash...)), //h.Sum(nil),
			}
			node.Parent = parentNode
			node = parentNode
			lastNode = lastNode.Parent
		}
		if n == 1 {
			//h := m.hashStrategy()
			//if _, err := h.Write(append(lastNode.Hash, node.Hash...)); err != nil {
			//	return err
			//}
			root := &Node{
				Tree:  m,
				Left:  lastNode,
				Right: node,
				Hash:  m.safeHash(append(lastNode.Hash, node.Hash...)), //h.Sum(nil),
				C:     nil,
			}
			node.Parent = root
			lastNode.Parent = root
			m.Root = root
		} else {
			node.Parent = lastNode.Parent
			lastNode.Parent.Right = node
			for ; node.Parent != nil; node = node.Parent {
				//h := m.hashStrategy()
				//if _, err := h.Write(append(node.Parent.Left.Hash, node.Hash...)); err != nil {
				//	return err
				//}
				node.Parent.Hash = m.safeHash(append(node.Parent.Left.Hash, node.Hash...)) //h.Sum(nil)
			}
		}
	}
	m.merkleRoot = m.Root.Hash
	return nil
}

// GetMerklePath: Get Merkle path and indexes(left leaf or right leaf)
func (m *MerkleTree) GetMerklePath(content Content) ([][]byte, []int64, error) {
	for _, current := range m.Leafs {
		ok, err := current.C.Equals(content)
		if err != nil {
			return nil, nil, err
		}

		if ok {
			currentParent := current.Parent
			var merklePath [][]byte
			var index []int64
			for currentParent != nil {
				if bytes.Equal(currentParent.Left.Hash, current.Hash) {
					merklePath = append(merklePath, currentParent.Right.Hash)
					index = append(index, 1) // right leaf
				} else {
					merklePath = append(merklePath, currentParent.Left.Hash)
					index = append(index, 0) // left leaf
				}
				current = currentParent
				currentParent = currentParent.Parent
			}
			return merklePath, index, nil
		}
	}
	return nil, nil, nil
}

// buildWithContent is a helper function that for a given set of Contents, generates a
// corresponding tree and returns the root node, a list of leaf nodes, and a possible error.
// Returns an error if cs contains no Contents.
func buildWithContent(cs []Content, t *MerkleTree) (*Node, []*Node, error) {
	if len(cs) == 0 {
		return nil, nil, errors.New("error: cannot construct tree with no content")
	}
	var leafs []*Node
	for _, c := range cs {
		hash, err := c.CalculateHash()
		if err != nil {
			return nil, nil, err
		}

		leafs = append(leafs, &Node{
			Hash: hash,
			C:    c,
			leaf: true,
			Tree: t,
		})
	}
	if len(leafs)%2 == 1 {
		duplicate := &Node{
			Hash: leafs[len(leafs)-1].Hash,
			C:    leafs[len(leafs)-1].C,
			leaf: true,
			dup:  true,
			Tree: t,
		}
		leafs = append(leafs, duplicate)
	}
	root, err := buildIntermediate(leafs, t)
	if err != nil {
		return nil, nil, err
	}

	return root, leafs, nil
}

// buildIntermediate is a helper function that for a given list of leaf nodes, constructs
// the intermediate and root levels of the tree. Returns the resulting root node of the tree.
func buildIntermediate(nl []*Node, t *MerkleTree) (*Node, error) {
	var nodes []*Node
	for i := 0; i < len(nl); i += 2 {
		var left, right int = i, i + 1
		if i+1 == len(nl) {
			right = i
		}
		//h := t.hashStrategy()
		//if _, err := h.Write(append(nl[left].Hash, nl[right].Hash...)); err != nil {
		//	return nil, err
		//}
		n := &Node{
			Left:  nl[left],
			Right: nl[right],
			Hash:  t.safeHash(append(nl[left].Hash, nl[right].Hash...)), //h.Sum(nil),
			Tree:  t,
		}
		nodes = append(nodes, n)
		nl[left].Parent = n
		nl[right].Parent = n
		if len(nl) == 2 {
			return n, nil
		}
	}
	return buildIntermediate(nodes, t)
}

// MerkleRoot returns the unverified Merkle Root (hash of the root node) of the tree.
func (m *MerkleTree) MerkleRoot() []byte {
	return m.merkleRoot
}

// RebuildTree is a helper function that will rebuild the tree reusing only the content that
// it holds in the leaves.
func (m *MerkleTree) RebuildTree() error {
	var cs []Content
	for _, c := range m.Leafs {
		cs = append(cs, c.C)
	}
	root, leafs, err := buildWithContent(cs, m)
	if err != nil {
		return err
	}
	m.Root = root
	m.Leafs = leafs
	m.merkleRoot = root.Hash
	return nil
}

// RebuildTreeWith replaces the content of the tree and does a complete rebuild; while the root of
// the tree will be replaced the MerkleTree completely survives this operation. Returns an error if the
// list of content cs contains no entries.
func (m *MerkleTree) RebuildTreeWith(cs []Content) error {
	root, leafs, err := buildWithContent(cs, m)
	if err != nil {
		return err
	}
	m.Root = root
	m.Leafs = leafs
	m.merkleRoot = root.Hash
	return nil
}

// VerifyTree verify tree validates the hashes at each level of the tree and returns true if the
// resulting hash at the root of the tree matches the resulting root hash; returns false otherwise.
func (m *MerkleTree) VerifyTree() (bool, error) {
	calculatedMerkleRoot, err := m.Root.verifyNode()
	if err != nil {
		return false, err
	}

	//if bytes.Compare(m.merkleRoot, calculatedMerkleRoot) == 0 {
	if bytes.Equal(m.merkleRoot, calculatedMerkleRoot) {
		return true, nil
	}
	return false, nil
}

// VerifyContent indicates whether a given content is in the tree and the hashes are valid for that content.
// Returns true if the expected Merkle Root is equivalent to the Merkle root calculated on the critical path
// for a given content. Returns true if valid and false otherwise.
func (m *MerkleTree) VerifyContent(content Content) (bool, error) {
	for _, l := range m.Leafs {
		ok, err := l.C.Equals(content)
		if err != nil {
			return false, err
		}

		if ok {
			currentParent := l.Parent
			for currentParent != nil {
				rightBytes, err := currentParent.Right.calculateNodeHash()
				if err != nil {
					return false, err
				}

				leftBytes, err := currentParent.Left.calculateNodeHash()
				if err != nil {
					return false, err
				}
				//h := m.hashStrategy()
				//if _, err := h.Write(append(leftBytes, rightBytes...)); err != nil {
				//	return false, err
				//}
				if !bytes.Equal(m.safeHash(append(leftBytes, rightBytes...)), currentParent.Hash) {
					return false, nil
				}
				currentParent = currentParent.Parent
			}
			return true, nil
		}
	}
	return false, nil
}

// String returns a string representation of the node.
func (n *Node) String() string {
	return fmt.Sprintf("%t %t %v %s", n.leaf, n.dup, n.Hash, n.C)
}

// String returns a string representation of the tree. Only leaf nodes are included
// in the output.
func (m *MerkleTree) String() string {
	s := ""
	for _, l := range m.Leafs {
		s += fmt.Sprint(l)
		s += "\n"
	}
	return s
}

func print2DUtil(root *Node, space int) {
	if root == nil {
		return
	}
	// Increase distance between levels
	space += 9
	// Process right child first
	print2DUtil(root.Right, space)
	// Print current node after space
	// count
	//fmt.Println()
	for i := 5; i < space; i++ {
		if i == space-6 {
			fmt.Print("|")
		}
		if i > space-7 && space > 9 {
			fmt.Print("-")
		} else {
			fmt.Print(" ")
		}
	}
	fmt.Print("[")
	fmt.Print(root.Hash[31])
	fmt.Println("]")
	// Process left child
	print2DUtil(root.Left, space)
}

func prettyPrint(root *Node, space int) {
	print2DUtil(root, space)
	for i := 0; i < 4; i++ {
		fmt.Println("")
	}
}
