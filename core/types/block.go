// Copyright 2018 The go-ethereum Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

// Package types contains data types related to Cortex consensus.
package types

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math/big"
	"reflect"
	"slices"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

// A BlockNonce is a 64-bit hash which proves (combined with the
// mix-hash) that a sufficient amount of computation has been carried
// out on a block.
type BlockNonce [8]byte
type BlockSolution [42]uint32

// EncodeNonce converts the given integer to a block nonce.
func EncodeNonce(i uint64) BlockNonce {
	var n BlockNonce
	binary.BigEndian.PutUint64(n[:], i)
	return n
}

// Uint64 returns the integer value of a block nonce.
func (n BlockNonce) Uint64() uint64 {
	return binary.BigEndian.Uint64(n[:])
}

// MarshalText encodes n as a hex string with 0x prefix.
func (n BlockNonce) MarshalText() ([]byte, error) {
	return hexutil.Bytes(n[:]).MarshalText()
}

// UnmarshalText implements encoding.TextUnmarshaler.
func (n *BlockNonce) UnmarshalText(input []byte) error {
	return hexutil.UnmarshalFixedText("BlockNonce", input, n[:])
}

func (s *BlockSolution) Uint32() []uint32 { return s[:] }

func (s *BlockSolution) MarshalText() ([]byte, error) {
	buf := make([]byte, 42*4)
	//for i := 0; i < len(s); i++ {
	for i, sol := range s {
		binary.BigEndian.PutUint32(buf[i*4:], sol)
	}
	return buf, nil
	//buf := new(bytes.Buffer)
	//for _, d := range s {
	//	binary.Write(buf, binary.BigEndian, d)
	//}
	//return buf.Bytes(), nil
}

func (s *BlockSolution) UnmarshalText(input []byte) error {
	for i := 0; i < 42; i++ {
		s[i] = binary.BigEndian.Uint32(input[i*4 : i*4+4])
	}
	//buf := bytes.NewBuffer(input)
	//for i := 0; i < 42 && buf.Len() > 3; i++ {
	//	s[i] = binary.BigEndian.Uint32(buf.Next(4))
	//}
	return nil
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *BlockSolution) UnmarshalJSON(input []byte) error {
	for i := 0; i < 42; i++ {
		s[i] = uint32(input[i])
	}
	return nil
}

//go:generate go run github.com/fjl/gencodec -type Header -field-override headerMarshaling -out gen_header_json.go

// Header represents a block header in the Cortex blockchain.
type Header struct {
	ParentHash  common.Hash    `json:"parentHash"       gencodec:"required"`
	UncleHash   common.Hash    `json:"sha3Uncles"       gencodec:"required"`
	Coinbase    common.Address `json:"miner"`
	Root        common.Hash    `json:"stateRoot"        gencodec:"required"`
	TxHash      common.Hash    `json:"transactionsRoot" gencodec:"required"`
	ReceiptHash common.Hash    `json:"receiptsRoot"     gencodec:"required"`
	Bloom       Bloom          `json:"logsBloom"        gencodec:"required"`
	Difficulty  *big.Int       `json:"difficulty"       gencodec:"required"`
	Number      *big.Int       `json:"number"           gencodec:"required"`
	GasLimit    uint64         `json:"gasLimit"         gencodec:"required"`
	GasUsed     uint64         `json:"gasUsed"          gencodec:"required"`
	Time        uint64         `json:"timestamp"        gencodec:"required"`
	Extra       []byte         `json:"extraData"        gencodec:"required"`
	MixDigest   common.Hash    `json:"mixHash"`
	Nonce       BlockNonce     `json:"nonce"`
	Solution    BlockSolution  `json:"solution"`
	Quota       uint64         `json:"quota"            gencodec:"required"`
	QuotaUsed   uint64         `json:"quotaUsed"        gencodec:"required"`
	Supply      *big.Int       `json:"supply"           gencodec:"required"`
}

// field type overrides for gencodec
type headerMarshaling struct {
	Difficulty *hexutil.Big
	Number     *hexutil.Big
	GasLimit   hexutil.Uint64
	GasUsed    hexutil.Uint64
	Time       hexutil.Uint64
	Extra      hexutil.Bytes
	Hash       common.Hash `json:"hash"` // adds call to Hash() in MarshalJSON
	Quota      hexutil.Uint64
	QuotaUsed  hexutil.Uint64
	Supply     *hexutil.Big
}

// Hash returns the block hash of the header, which is simply the keccak256 hash of its
// RLP encoding.
func (h *Header) Hash() common.Hash {
	return rlpHash(h)
}

var headerSize = common.StorageSize(reflect.TypeOf(Header{}).Size())

// Size returns the approximate memory used by all internal contents. It is used
// to approximate and limit the memory consumption of various caches.
func (h *Header) Size() common.StorageSize {
	//return common.StorageSize(unsafe.Sizeof(*h)) + common.StorageSize(len(h.Extra)+(h.Difficulty.BitLen()+h.Number.BitLen()+h.Time.BitLen())/8)
	return headerSize + common.StorageSize(len(h.Extra)+(h.Difficulty.BitLen()+h.Number.BitLen())/8)
}

func (h *Header) SanityCheck() error {
	if h.Number != nil && !h.Number.IsUint64() {
		return fmt.Errorf("too large block number: bitlen %d", h.Number.BitLen())
	}
	if h.Difficulty != nil {
		if diffLen := h.Difficulty.BitLen(); diffLen > 80 {
			return fmt.Errorf("too large block difficulty: bitlen %d", diffLen)
		}
	}
	if eLen := len(h.Extra); eLen > 100*1024 {
		return fmt.Errorf("too large block extradata: size %d", eLen)
	}
	return nil
}

// EmptyBody returns true if there is no additional 'body' to complete the header
// that is: no transactions and no uncles.
func (h *Header) EmptyBody() bool {
	return h.TxHash == EmptyRootHash && h.UncleHash == EmptyUncleHash
}

// EmptyReceipts returns true if there are no receipts for this header/block.
func (h *Header) EmptyReceipts() bool {
	return h.ReceiptHash == EmptyRootHash
}

// Body is a simple (mutable, non-safe) data container for storing and moving
// a block's data contents (transactions and uncles) tocortexer.
type Body struct {
	Transactions []*Transaction
	Uncles       []*Header
}

// Block represents an entire block in the Cortex blockchain.
type Block struct {
	header       *Header
	uncles       []*Header
	transactions Transactions

	// caches
	hash atomic.Pointer[common.Hash]
	size atomic.Uint64

	// Td is used by package core to store the total difficulty
	// of the chain up to and including the block.
	td *big.Int

	// These fields are used by package ctxc to track
	// inter-peer block relay.
	ReceivedAt   time.Time
	ReceivedFrom any
}

// DeprecatedTd is an old relic for extracting the TD of a block. It is in the
// code solely to facilitate upgrading the database from the old format to the
// new, after which it should be deleted. Do not use!
func (b *Block) DeprecatedTd() *big.Int {
	return b.td
}

// [deprecated by ctxc/63]
// StorageBlock defines the RLP encoding of a Block stored in the
// state database. The StorageBlock encoding contains fields that
// would otherwise need to be recomputed.
type StorageBlock Block

// "external" block encoding. used for ctxc protocol, etc.
type extblock struct {
	Header *Header
	Txs    []*Transaction
	Uncles []*Header
}

// [deprecated by ctxc/63]
// "storage" block encoding. used for database.
type storageblock struct {
	Header *Header
	Txs    []*Transaction
	Uncles []*Header
	TD     *big.Int
}

// NewBlock creates a new block. The input data is copied,
// changes to header and to the field values will not affect the
// block.
//
// relevant portions of the header.
//
// The receipt's bloom must already calculated for the block's bloom to be
// correctly calculated.
func NewBlock(header *Header, body *Body, receipts []*Receipt, hasher ListHasher) *Block {
	if body == nil {
		body = &Body{}
	}
	var (
		b      = NewBlockWithHeader(header)
		txs    = body.Transactions
		uncles = body.Uncles
	)

	if len(txs) == 0 {
		b.header.TxHash = EmptyRootHash
	} else {
		b.header.TxHash = DeriveSha(Transactions(txs), hasher)
		b.transactions = make(Transactions, len(txs))
		copy(b.transactions, txs)
	}

	if len(receipts) == 0 {
		b.header.ReceiptHash = EmptyRootHash
	} else {
		b.header.ReceiptHash = DeriveSha(Receipts(receipts), hasher)
		// Receipts must go through MakeReceipt to calculate the receipt's bloom
		// already. Merge the receipt's bloom together instead of recalculating
		// everything.
		b.header.Bloom = MergeBloom(receipts)
	}

	if len(uncles) == 0 {
		b.header.UncleHash = EmptyUncleHash
	} else {
		b.header.UncleHash = CalcUncleHash(uncles)
		b.uncles = make([]*Header, len(uncles))
		for i := range uncles {
			b.uncles[i] = CopyHeader(uncles[i])
		}
	}

	return b
}

// CopyHeader creates a deep copy of a block header to prevent side effects from
// modifying a header variable.
func CopyHeader(h *Header) *Header {
	cpy := *h
	//if cpy.Time = new(big.Int); h.Time != nil {
	//	cpy.Time.Set(h.Time)
	//}
	if cpy.Difficulty = new(big.Int); h.Difficulty != nil {
		cpy.Difficulty.Set(h.Difficulty)
	}
	if cpy.Number = new(big.Int); h.Number != nil {
		cpy.Number.Set(h.Number)
	}
	if len(h.Extra) > 0 {
		cpy.Extra = make([]byte, len(h.Extra))
		copy(cpy.Extra, h.Extra)
	}
	if cpy.Supply = new(big.Int); h.Supply != nil {
		cpy.Supply.Set(h.Supply)
	}
	return &cpy
}

// DecodeRLP decodes the Cortex
func (b *Block) DecodeRLP(s *rlp.Stream) error {
	var eb extblock
	_, size, _ := s.Kind()
	if err := s.Decode(&eb); err != nil {
		return err
	}
	b.header, b.uncles, b.transactions = eb.Header, eb.Uncles, eb.Txs
	b.size.Store(rlp.ListSize(size))
	return nil
}

// EncodeRLP serializes b into the Cortex RLP block format.
func (b *Block) EncodeRLP(w io.Writer) error {
	return rlp.Encode(w, extblock{
		Header: b.header,
		Txs:    b.transactions,
		Uncles: b.uncles,
	})
}

// [deprecated by ctxc/63]
func (b *StorageBlock) DecodeRLP(s *rlp.Stream) error {
	var sb storageblock
	if err := s.Decode(&sb); err != nil {
		return err
	}
	b.header, b.uncles, b.transactions, b.td = sb.Header, sb.Uncles, sb.Txs, sb.TD
	return nil
}

// TODO: copies

func (b *Block) Uncles() []*Header          { return b.uncles }
func (b *Block) Transactions() Transactions { return b.transactions }

func (b *Block) Transaction(hash common.Hash) *Transaction {
	for _, transaction := range b.transactions {
		if transaction.Hash() == hash {
			return transaction
		}
	}
	return nil
}

func (b *Block) Number() *big.Int     { return new(big.Int).Set(b.header.Number) }
func (b *Block) GasLimit() uint64     { return b.header.GasLimit }
func (b *Block) GasUsed() uint64      { return b.header.GasUsed }
func (b *Block) Difficulty() *big.Int { return new(big.Int).Set(b.header.Difficulty) }
func (b *Block) Time() uint64         { return b.header.Time }

func (b *Block) NumberU64() uint64        { return b.header.Number.Uint64() }
func (b *Block) MixDigest() common.Hash   { return b.header.MixDigest }
func (b *Block) Nonce() uint64            { return binary.BigEndian.Uint64(b.header.Nonce[:]) }
func (b *Block) Solution() BlockSolution  { return b.header.Solution }
func (b *Block) Bloom() Bloom             { return b.header.Bloom }
func (b *Block) Coinbase() common.Address { return b.header.Coinbase }
func (b *Block) Root() common.Hash        { return b.header.Root }
func (b *Block) ParentHash() common.Hash  { return b.header.ParentHash }
func (b *Block) TxHash() common.Hash      { return b.header.TxHash }
func (b *Block) ReceiptHash() common.Hash { return b.header.ReceiptHash }
func (b *Block) UncleHash() common.Hash   { return b.header.UncleHash }
func (b *Block) Extra() []byte            { return common.CopyBytes(b.header.Extra) }
func (b *Block) Quota() uint64            { return b.header.Quota }
func (b *Block) QuotaUsed() uint64        { return b.header.QuotaUsed }
func (b *Block) Supply() *big.Int         { return new(big.Int).Set(b.header.Supply) }
func (b *Block) Header() *Header          { return CopyHeader(b.header) }

// Body returns the non-header content of the block.
func (b *Block) Body() *Body { return &Body{b.transactions, b.uncles} }

// Size returns the true RLP encoded storage size of the block, either by encoding
// and returning it, or returning a previsouly cached value.
func (b *Block) Size() uint64 {
	if size := b.size.Load(); size > 0 {
		return size
	}
	c := writeCounter(0)
	rlp.Encode(&c, b)
	b.size.Store(uint64(c))
	return uint64(c)
}

func (b *Block) SanityCheck() error {
	return b.header.SanityCheck()
}

type writeCounter uint64

func (c *writeCounter) Write(b []byte) (int, error) {
	*c += writeCounter(len(b))
	return len(b), nil
}

func CalcUncleHash(uncles []*Header) common.Hash {
	return rlpHash(uncles)
}

// CalcRequestsHash creates the block requestsHash value for a list of requests.
func CalcRequestsHash(requests [][]byte) common.Hash {
	h1, h2 := sha256.New(), sha256.New()
	var buf common.Hash
	for _, item := range requests {
		if len(item) > 1 { // skip items with only requestType and no data.
			h1.Reset()
			h1.Write(item)
			h2.Write(h1.Sum(buf[:0]))
		}
	}
	h2.Sum(buf[:0])
	return buf
}

// NewBlockWithHeader creates a block with the given header data. The
// header data is copied, changes to header and to the field values
// will not affect the block.
func NewBlockWithHeader(header *Header) *Block {
	return &Block{header: CopyHeader(header)}
}

// WithSeal returns a new block with the data from b but the header replaced with
// the sealed one.
func (b *Block) WithSeal(header *Header) *Block {
	cpy := *header

	return &Block{
		header:       &cpy,
		transactions: b.transactions,
		uncles:       b.uncles,
	}
}

// WithBody returns a new block with the original header and a deep copy of the
// provided body.
func (b *Block) WithBody(body Body) *Block {
	block := &Block{
		header:       b.header,
		transactions: slices.Clone(body.Transactions),
		uncles:       make([]*Header, len(body.Uncles)),
	}
	for i := range body.Uncles {
		block.uncles[i] = CopyHeader(body.Uncles[i])
	}
	return block
}

// Hash returns the keccak256 hash of b's header.
// The hash is computed on the first call and cached thereafter.
func (b *Block) Hash() common.Hash {
	if hash := b.hash.Load(); hash != nil {
		return *hash
	}
	h := b.header.Hash()
	b.hash.Store(&h)
	return h
}

type Blocks []*Block

// HeaderParentHashFromRLP returns the parentHash of an RLP-encoded
// header. If 'header' is invalid, the zero hash is returned.
func HeaderParentHashFromRLP(header []byte) common.Hash {
	// parentHash is the first list element.
	listContent, _, err := rlp.SplitList(header)
	if err != nil {
		return common.Hash{}
	}
	parentHash, _, err := rlp.SplitString(listContent)
	if err != nil {
		return common.Hash{}
	}
	if len(parentHash) != 32 {
		return common.Hash{}
	}
	return common.BytesToHash(parentHash)
}

/*type BlockBy func(b1, b2 *Block) bool

func (self BlockBy) Sort(blocks Blocks) {
	bs := blockSorter{
		blocks: blocks,
		by:     self,
	}
	sort.Sort(bs)
}

type blockSorter struct {
	blocks Blocks
	by     func(b1, b2 *Block) bool
}

func (self blockSorter) Len() int { return len(self.blocks) }
func (self blockSorter) Swap(i, j int) {
	self.blocks[i], self.blocks[j] = self.blocks[j], self.blocks[i]
}
func (self blockSorter) Less(i, j int) bool { return self.by(self.blocks[i], self.blocks[j]) }

func Number(b1, b2 *Block) bool { return b1.header.Number.Cmp(b2.header.Number) < 0 }*/
