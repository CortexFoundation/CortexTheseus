package ctxc

import (
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

const (
	StatusMsg          = 0x00
	NewBlockHashesMsg  = 0x01
	TransactionMsg     = 0x02
	GetBlockHeadersMsg = 0x03
	BlockHeadersMsg    = 0x04
	GetBlockBodiesMsg  = 0x05
	BlockBodiesMsg     = 0x06
	NewBlockMsg        = 0x07
	GetNodeDataMsg     = 0x0d
	NodeDataMsg        = 0x0e
	GetReceiptsMsg     = 0x0f
	ReceiptsMsg        = 0x10
	// New protocol message codes introduced in eth65
	//
	// Previously these message ids(0x08, 0x09) were used by some
	// legacy and unsupported eth protocols, reown them here.
	NewPooledTransactionHashesMsg = 0x08
	GetPooledTransactionsMsg      = 0x09
	PooledTransactionsMsg         = 0x0a
)

// BlockHeadersRLPPacket represents a block header response, to use when we already
// have the headers rlp encoded.
type BlockHeadersRLPPacket []rlp.RawValue

// BlockHeadersPacket represents a block header response over eth/66.
type BlockHeadersRLPPacket66 struct {
	RequestId uint64
	BlockHeadersRLPPacket
}
