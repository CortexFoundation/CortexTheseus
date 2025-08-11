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
	return n.Tree.safeHash(append(leftBytes, rightBytes...)), nil
}

// calculateNodeHash is a helper function that calculates the hash of the node.
func (n *Node) calculateNodeHash() ([]byte, error) {
	if n.leaf {
		return n.C.CalculateHash()
	}
	return n.Tree.safeHash(append(n.Left.Hash, n.Right.Hash...)), nil
}

// NewTree creates a new Merkle Tree using the content cs.
func NewTree(cs []Content) (*MerkleTree, error) {
	t := &MerkleTree{}
	root, leafs, err := buildWithContent(cs, t)
	if err != nil {
		return nil, err
	}
	t.Root = root
	t.Leafs = leafs
	t.merkleRoot = root.Hash
	return t, nil
}

func (m *MerkleTree) safeHash(data []byte) []byte {
	h := newHasher()
	defer returnHasherToPool(h)
	return h.sum(data)
}

// AddNode adds a new node to the Merkle Tree, updating the tree structure as needed.
func (m *MerkleTree) AddNode(c Content) error {
	hash, err := c.CalculateHash()
	if err != nil {
		return err
	}
	n := len(m.Leafs)

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
		combinedHash := append(newLeaf.Hash, dupLeaf.Hash...)
		root := &Node{
			Tree:  m,
			Left:  newLeaf,
			Right: dupLeaf,
			Hash:  m.safeHash(combinedHash),
			C:     nil,
		}
		newLeaf.Parent = root
		dupLeaf.Parent = root
		m.Root = root
		m.merkleRoot = root.Hash
		m.Leafs = []*Node{newLeaf, dupLeaf}
		return nil
	}

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
			combinedHash := append(newLeaf.Parent.Left.Hash, newLeaf.Hash...)
			newLeaf.Parent.Hash = m.safeHash(combinedHash)
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
		m.Leafs = append(m.Leafs, newLeaf, dupLeaf)

		combinedHash := append(newLeaf.Hash, dupLeaf.Hash...)
		node := &Node{
			Tree:  m,
			Left:  newLeaf,
			Right: dupLeaf,
			Hash:  m.safeHash(combinedHash),
		}
		newLeaf.Parent = node
		dupLeaf.Parent = node
		lastNode := m.Leafs[n-1].Parent

		for n /= 2; n%2 == 0; n /= 2 {
			combinedHash = append(node.Hash, node.Hash...)
			parentNode := &Node{
				Tree:  m,
				Left:  node,
				Right: node,
				Hash:  m.safeHash(combinedHash),
			}
			node.Parent = parentNode
			node = parentNode
			lastNode = lastNode.Parent
		}
		if n == 1 {
			combinedHash = append(lastNode.Hash, node.Hash...)
			root := &Node{
				Tree:  m,
				Left:  lastNode,
				Right: node,
				Hash:  m.safeHash(combinedHash),
				C:     nil,
			}
			node.Parent = root
			lastNode.Parent = root
			m.Root = root
		} else {
			node.Parent = lastNode.Parent
			lastNode.Parent.Right = node
			for ; node.Parent != nil; node = node.Parent {
				combinedHash = append(node.Parent.Left.Hash, node.Hash...)
				node.Parent.Hash = m.safeHash(combinedHash)
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
			merklePath := make([][]byte, 0, 16) // Pre-allocate with a reasonable capacity
			index := make([]int64, 0, 16)
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
	leafs := make([]*Node, 0, len(cs))
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
	if len(nl) == 1 {
		return nl[0], nil
	}

	nodes := make([]*Node, 0, (len(nl)+1)/2)
	for i := 0; i < len(nl); i += 2 {
		left, right := i, i+1
		if i+1 == len(nl) {
			right = i
		}

		combinedHashBytes := make([]byte, 0, len(nl[left].Hash)+len(nl[right].Hash))
		combinedHashBytes = append(combinedHashBytes, nl[left].Hash...)
		combinedHashBytes = append(combinedHashBytes, nl[right].Hash...)

		n := &Node{
			Left:  nl[left],
			Right: nl[right],
			Hash:  t.safeHash(combinedHashBytes),
			Tree:  t,
		}
		nodes = append(nodes, n)
		nl[left].Parent = n
		nl[right].Parent = n
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
	cs := make([]Content, 0, len(m.Leafs))
	for _, c := range m.Leafs {
		if !c.dup {
			cs = append(cs, c.C)
		}
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
	if m.Root == nil {
		return false, errors.New("tree is empty")
	}
	calculatedMerkleRoot, err := m.Root.verifyNode()
	if err != nil {
		return false, err
	}
	return bytes.Equal(m.merkleRoot, calculatedMerkleRoot), nil
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
				leftHash := currentParent.Left.Hash
				rightHash := currentParent.Right.Hash

				combinedHash := make([]byte, 0, len(leftHash)+len(rightHash))
				combinedHash = append(combinedHash, leftHash...)
				combinedHash = append(combinedHash, rightHash...)

				if !bytes.Equal(m.safeHash(combinedHash), currentParent.Hash) {
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
	return fmt.Sprintf("leaf: %t, dup: %t, hash: %x, content: %v", n.leaf, n.dup, n.Hash, n.C)
}

// String returns a string representation of the tree. Only leaf nodes are included
// in the output.
func (m *MerkleTree) String() string {
	s := ""
	for _, l := range m.Leafs {
		s += fmt.Sprintf("%s\n", l.String())
	}
	return s
}

func (m *MerkleTree) Purge() {
	m.Root = nil
	m.merkleRoot = nil
	m.Leafs = nil
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
