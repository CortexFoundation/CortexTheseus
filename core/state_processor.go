// Copyright 2019 The go-ethereum Authors
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

package core

import (
	"fmt"
	"math/big"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/core/state"
	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/core/vm"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/params"
)

// StateProcessor is a basic Processor, which takes care of transitioning
// state from one point to another.
//
// StateProcessor implements Processor.
type StateProcessor struct {
	chain ChainContext // Chain context interface
}

// NewStateProcessor initialises a new StateProcessor.
func NewStateProcessor(chain ChainContext) *StateProcessor {
	return &StateProcessor{
		chain: chain,
	}
}

// chainConfig returns the chain configuration.
func (p *StateProcessor) chainConfig() *params.ChainConfig {
	return p.chain.Config()
}

// Process processes the state changes according to the Cortex rules by running
// the transaction messages using the statedb and applying any rewards to both
// the processor (coinbase) and any included uncles.
//
// Process returns the receipts and logs accumulated during the process and
// returns the amount of gas that was used in the process. If any of the
// transactions failed to execute due to insufficient gas it will return an error.
func (p *StateProcessor) Process(block *types.Block, statedb *state.StateDB, cfg vm.Config) (*ProcessResult, error) {
	var (
		config   = p.chainConfig()
		receipts types.Receipts
		usedGas  = new(uint64)
		//quotaLimit = big.NewInt(0)//parent.quota+64k - parent.quotaUsed
		header      = block.Header()
		blockHash   = block.Hash()
		blockNumber = block.Number()
		allLogs     []*types.Log
		gp          = new(GasPool).AddGas(block.GasLimit())
		qp          = NewQuotaPool(header.Quota)
	)
	if err := qp.SubQuota(header.QuotaUsed); err != nil {
		return nil, err
	}
	//*usedQuota = quotaUsed
	// Mutate the block and state according to any hard-fork specs
	//if p.config.DAOForkSupport && p.config.DAOForkBlock != nil && p.config.DAOForkBlock.Cmp(block.Number()) == 0 {
	//	misc.ApplyDAOHardFork(statedb)
	//}
	var (
		blockContext = NewCVMBlockContext(header, p.chain, nil)
		vmenv        = vm.NewCVM(blockContext, statedb, config, cfg)
		signer       = types.MakeSigner(config, header.Number, header.Time)
	)
	// Iterate over and process the individual transactions
	for i, tx := range block.Transactions() {
		msg, err := TransactionToMessage(tx, signer)
		if err != nil {
			return nil, fmt.Errorf("could not apply tx %d [%v]: %w", i, tx.Hash().Hex(), err)
		}
		statedb.SetTxContext(tx.Hash(), i)
		receipt, err := applyTransaction(msg, config, gp, qp, statedb, header, blockNumber, blockHash, tx, usedGas, vmenv)
		if err != nil {
			return nil, fmt.Errorf("could not apply tx %d [%v]: %w", i, tx.Hash().Hex(), err)
		}
		receipts = append(receipts, receipt)
		allLogs = append(allLogs, receipt.Logs...)
	}

	//parent:=p.bc.GetHeaderByHash(header.ParentHash)
	//if parent == nil {
	//        return nil,nil,0,consensus.ErrUnknownAncestor
	//}
	// Finalize the block, applying any consensus engine specific extras (e.g. block rewards)
	p.chain.Engine().Finalize(p.chain, header, statedb, block.Transactions(), block.Uncles())

	return &ProcessResult{
		Receipts: receipts,
		Requests: nil,
		Logs:     allLogs,
		GasUsed:  *usedGas,
	}, nil
}

// ApplyTransaction attempts to apply a transaction to the given state database
// and uses the input parameters for its environment. It returns the receipt
// for the transaction, gas used and an error if the transaction failed,
// indicating the block was invalid.
func applyTransaction(msg *Message, config *params.ChainConfig, gp *GasPool, qp *QuotaPool, statedb *state.StateDB, header *types.Header, blockNumber *big.Int, blockHash common.Hash, tx *types.Transaction, usedGas *uint64, cvm *vm.CVM) (*types.Receipt, error) {
	// Create a new context to be used in the CVM environment
	txContext := NewCVMTxContext(msg)
	cvm.Reset(txContext, statedb)
	// Apply the transaction to the current state (included in the env)
	result, err := ApplyMessage(cvm, msg, gp, qp)
	if err != nil {
		return nil, err
	}

	if result.Quota > 0 {
		header.QuotaUsed += result.Quota

		if header.Quota < header.QuotaUsed {
			header.QuotaUsed -= result.Quota
			return nil, ErrQuotaLimitReached //errors.New("quota")
		}
	}

	// Update the state with pending changes
	var root []byte
	if config.IsByzantium(header.Number) {
		statedb.Finalise(true)
	} else {
		root = statedb.IntermediateRoot(config.IsEIP158(header.Number)).Bytes()
	}

	*usedGas += result.UsedGas
	return MakeReceipt(cvm, result, statedb, blockNumber, blockHash, tx, *usedGas, root), nil
}

// MakeReceipt generates the receipt object for a transaction given its execution result.
func MakeReceipt(cvm *vm.CVM, result *ExecutionResult, statedb *state.StateDB, blockNumber *big.Int, blockHash common.Hash, tx *types.Transaction, usedGas uint64, root []byte) *types.Receipt {
	// Create a new receipt for the transaction, storing the intermediate root and gas used
	// by the tx.
	receipt := types.NewReceipt(root, result.Failed(), usedGas)
	receipt.TxHash = tx.Hash()
	receipt.GasUsed = result.UsedGas
	// if the transaction created a contract, store the creation address in the receipt.
	if tx.To() == nil {
		receipt.ContractAddress = crypto.CreateAddress(cvm.TxContext.Origin, tx.Nonce())
	}
	// Set the receipt logs and create a bloom for filtering
	receipt.Logs = statedb.GetLogs(tx.Hash(), blockNumber.Uint64(), blockHash)
	receipt.Bloom = types.CreateBloom(receipt)
	receipt.BlockHash = blockHash
	receipt.BlockNumber = blockNumber
	receipt.TransactionIndex = uint(statedb.TxIndex())
	return receipt
}

// ApplyTransaction attempts to apply a transaction to the given state database
// and uses the input parameters for its environment. It returns the receipt
// for the transaction, gas used and an error if the transaction failed,
// indicating the block was invalid.
func ApplyTransaction(config *params.ChainConfig, bc ChainContext, author *common.Address, gp *GasPool, qp *QuotaPool, statedb *state.StateDB, header *types.Header, tx *types.Transaction, usedGas *uint64, cfg vm.Config) (*types.Receipt, error) {
	msg, err := TransactionToMessage(tx, types.MakeSigner(config, header.Number, header.Time))
	if err != nil {
		return nil, err
	}
	// Create a new context to be used in the CVM environment
	vmenv := vm.NewCVM(NewCVMBlockContext(header, bc, author), statedb, config, cfg)
	return applyTransaction(msg, config, gp, qp, statedb, header, header.Number, header.Hash(), tx, usedGas, vmenv)
}
