// +build !disable_miner

package miner

import (
	"math/big"
	"sync/atomic"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"

	"github.com/CortexFoundation/CortexTheseus/core/types"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
)

// commit runs any post-transaction state modifications, assembles the final block
// and commits new work if consensus engine is running.
func (w *worker) commit(uncles []*types.Header, interval func(), update bool, start time.Time) error {
	//w.mu.Lock()
	//defer w.mu.Unlock()
	// Deep copy receipts here to avoid interaction between different tasks.
	receipts := make([]*types.Receipt, len(w.current.receipts))
	for i, l := range w.current.receipts {
		receipts[i] = new(types.Receipt)
		*receipts[i] = *l
	}
	s := w.current.state.Copy()

	h := new(types.Header)
	*h = *w.current.header
	block, err := w.engine.Finalize(w.chain, h, s, w.current.txs, uncles, w.current.receipts)
	if err != nil {
		return err
	}
	if w.isRunning() {
		if interval != nil {
			interval()
		}
		select {
		case w.taskCh <- &task{receipts: receipts, state: s, block: block, createdAt: time.Now()}:
			w.unconfirmed.Shift(block.NumberU64() - 1)

			feesWei := new(big.Int)
			for i, tx := range block.Transactions() {
				feesWei.Add(feesWei, new(big.Int).Mul(new(big.Int).SetUint64(receipts[i].GasUsed), tx.GasPrice()))
			}
			feesCortex := new(big.Float).Quo(new(big.Float).SetInt(feesWei), new(big.Float).SetInt(big.NewInt(params.Cortex)))
			mined := new(big.Float).Quo(new(big.Float).SetInt(new(big.Int).Sub(block.Supply(), params.CTXC_INIT)), new(big.Float).SetInt(big.NewInt(params.Cortex)))
			peace := new(big.Float).Quo(new(big.Float).SetInt(block.Supply()), new(big.Float).SetInt(params.CTXC_TOP))
			capacity := new(big.Float).Quo(new(big.Float).SetInt(block.QuotaUsed()), new(big.Float).SetInt(block.Quota()))

			log.Info("Commit new mining work", "number", block.Number(), "sealhash", w.engine.SealHash(block.Header()),
				"uncles", len(uncles), "txs", w.current.tcount, "gas", block.GasUsed(), "fees", feesCortex, "elapsed", common.PrettyDuration(time.Since(start)), "diff", block.Difficulty(), "mined", mined, "peace", peace, "quota", block.Quota(), "used", block.QuotaUsed(), "capacity", capacity)

		case <-w.exitCh:
			log.Info("Worker has exited")
		}
	}
	if update {
		w.updateSnapshot()
	}
	return nil
}

func (self *Miner) Start(coinbase common.Address) {
	atomic.StoreInt32(&self.shouldStart, 1)
	self.SetCoinbase(coinbase)

	if atomic.LoadInt32(&self.canStart) == 0 {
		log.Info("Network syncing, will start miner afterwards")
		return
	}
	self.worker.start()
}
