package core

import "github.com/ethereum/go-ethereum/core/types"

// TxPreEvent is posted when a transaction enters the transaction pool.
type TxPreEvent struct{ Tx *types.Transaction }

// TxPostEvent is posted when a transaction has been processed.
type TxPostEvent struct{ Tx *types.Transaction }

// NewBlockEvent is posted when a block has been imported.
type NewBlockEvent struct{ Block *types.Block }

// NewMinedBlockEvent is posted when a block has been imported.
type NewMinedBlockEvent struct{ Block *types.Block }
