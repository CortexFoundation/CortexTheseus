package ethminer

import (
	"bytes"
	"github.com/ethereum/eth-go/ethchain"
	"github.com/ethereum/eth-go/ethlog"
	"github.com/ethereum/eth-go/ethreact"
	"github.com/ethereum/eth-go/ethwire"
	"sort"
)

var logger = ethlog.NewLogger("MINER")

type Miner struct {
	pow         ethchain.PoW
	ethereum    ethchain.EthManager
	coinbase    []byte
	reactChan   chan ethreact.Event
	txs         ethchain.Transactions
	uncles      []*ethchain.Block
	block       *ethchain.Block
	powChan     chan []byte
	powQuitChan chan ethreact.Event
	quitChan    chan bool
}

func NewDefaultMiner(coinbase []byte, ethereum ethchain.EthManager) Miner {
	miner := Miner{
		pow:      &ethchain.EasyPow{},
		ethereum: ethereum,
		coinbase: coinbase,
	}

	// Insert initial TXs in our little miner 'pool'
	miner.txs = ethereum.TxPool().Flush()
	miner.block = ethereum.BlockChain().NewBlock(miner.coinbase)

	return miner
}

func (miner *Miner) Start() {
	miner.reactChan = make(chan ethreact.Event, 1)   // This is the channel that receives 'updates' when ever a new transaction or block comes in
	miner.powChan = make(chan []byte, 1)             // This is the channel that receives valid sha hashes for a given block
	miner.powQuitChan = make(chan ethreact.Event, 1) // This is the channel that can exit the miner thread
	miner.quitChan = make(chan bool, 1)

	// Prepare inital block
	//miner.ethereum.StateManager().Prepare(miner.block.State(), miner.block.State())
	go miner.listener()

	reactor := miner.ethereum.Reactor()
	reactor.Subscribe("newBlock", miner.reactChan)
	reactor.Subscribe("newTx:pre", miner.reactChan)

	// We need the quit chan to be a Reactor event.
	// The POW search method is actually blocking and if we don't
	// listen to the reactor events inside of the pow itself
	// The miner overseer will never get the reactor events themselves
	// Only after the miner will find the sha
	reactor.Subscribe("newBlock", miner.powQuitChan)
	reactor.Subscribe("newTx:pre", miner.powQuitChan)

	logger.Infoln("Started")
}

func (miner *Miner) listener() {
out:
	for {
		select {
		case <-miner.quitChan:
			logger.Infoln("Stopped")
			break out
		case chanMessage := <-miner.reactChan:

			if block, ok := chanMessage.Resource.(*ethchain.Block); ok {
				//logger.Infoln("Got new block via Reactor")
				if bytes.Compare(miner.ethereum.BlockChain().CurrentBlock.Hash(), block.Hash()) == 0 {
					// TODO: Perhaps continue mining to get some uncle rewards
					//logger.Infoln("New top block found resetting state")

					// Filter out which Transactions we have that were not in this block
					var newtxs []*ethchain.Transaction
					for _, tx := range miner.txs {
						found := false
						for _, othertx := range block.Transactions() {
							if bytes.Compare(tx.Hash(), othertx.Hash()) == 0 {
								found = true
							}
						}
						if found == false {
							newtxs = append(newtxs, tx)
						}
					}
					miner.txs = newtxs

					// Setup a fresh state to mine on
					//miner.block = miner.ethereum.BlockChain().NewBlock(miner.coinbase, miner.txs)

				} else {
					if bytes.Compare(block.PrevHash, miner.ethereum.BlockChain().CurrentBlock.PrevHash) == 0 {
						logger.Infoln("Adding uncle block")
						miner.uncles = append(miner.uncles, block)
					}
				}
			}

			if tx, ok := chanMessage.Resource.(*ethchain.Transaction); ok {
				found := false
				for _, ctx := range miner.txs {
					if found = bytes.Compare(ctx.Hash(), tx.Hash()) == 0; found {
						break
					}

				}
				if found == false {
					// Undo all previous commits
					miner.block.Undo()
					// Apply new transactions
					miner.txs = append(miner.txs, tx)
				}
			}
		default:
			miner.mineNewBlock()
		}
	}
}

func (miner *Miner) Stop() {
	logger.Infoln("Stopping...")
	miner.quitChan <- true

	reactor := miner.ethereum.Reactor()
	reactor.Unsubscribe("newBlock", miner.powQuitChan)
	reactor.Unsubscribe("newTx:pre", miner.powQuitChan)
	reactor.Unsubscribe("newBlock", miner.reactChan)
	reactor.Unsubscribe("newTx:pre", miner.reactChan)

	close(miner.powQuitChan)
	close(miner.quitChan)
}

func (self *Miner) mineNewBlock() {
	stateManager := self.ethereum.StateManager()

	self.block = self.ethereum.BlockChain().NewBlock(self.coinbase)

	// Apply uncles
	if len(self.uncles) > 0 {
		self.block.SetUncles(self.uncles)
	}

	// Sort the transactions by nonce in case of odd network propagation
	sort.Sort(ethchain.TxByNonce{self.txs})

	// Accumulate all valid transactions and apply them to the new state
	// Error may be ignored. It's not important during mining
	parent := self.ethereum.BlockChain().GetBlock(self.block.PrevHash)
	coinbase := self.block.State().GetOrNewStateObject(self.block.Coinbase)
	coinbase.SetGasPool(self.block.CalcGasLimit(parent))
	receipts, txs, unhandledTxs, err := stateManager.ProcessTransactions(coinbase, self.block.State(), self.block, self.block, self.txs)
	if err != nil {
		logger.Debugln(err)
	}
	self.txs = append(txs, unhandledTxs...)

	// Set the transactions to the block so the new SHA3 can be calculated
	self.block.SetReceipts(receipts, txs)

	// Accumulate the rewards included for this block
	stateManager.AccumelateRewards(self.block.State(), self.block)

	self.block.State().Update()

	logger.Infof("Mining on block. Includes %v transactions", len(self.txs))

	// Find a valid nonce
	self.block.Nonce = self.pow.Search(self.block, self.powQuitChan)
	if self.block.Nonce != nil {
		err := self.ethereum.StateManager().Process(self.block, false)
		if err != nil {
			logger.Infoln(err)
		} else {
			self.ethereum.Broadcast(ethwire.MsgBlockTy, []interface{}{self.block.Value().Val})
			logger.Infof("🔨  Mined block %x\n", self.block.Hash())
			logger.Infoln(self.block)
			// Gather the new batch of transactions currently in the tx pool
			self.txs = self.ethereum.TxPool().CurrentTransactions()
		}
	}
}
