// Copyright 2023 The CortexTheseus Authors
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

package robot

import (
	"encoding/json"
	"errors"
	"math/big"
	"strconv"
	"time"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/mclock"
	"github.com/CortexFoundation/CortexTheseus/log"
	params1 "github.com/CortexFoundation/CortexTheseus/params"
	"github.com/CortexFoundation/torrentfs/types"
)

// solve block from node
func (m *Monitor) solve(block *types.Block) error {
	switch m.srv.Load() {
	case SRV_MODEL:
		return m.forModelService(block)
	//case 1:
	//	return m.forExplorerService(block) // others service, explorer, exchange, zkp, nft, etc.
	//case 2:
	//	return m.forExchangeService(block)
	case SRV_RECORD:
		return m.forRecordService(block)
	default:
		return errors.New("no block operation service found")
	}
}

func (m *Monitor) SwitchService(srv int) error {
	log.Debug("Srv switch start", "srv", srv, "ch", cap(m.srvCh))
	select {
	case m.srvCh <- srv:
	case <-m.exitCh:
		return nil
	}
	log.Debug("Srv switch end", "srv", srv, "ch", cap(m.srvCh))
	return nil
}

func (m *Monitor) doSwitch(srv int) error {
	if m.srv.Load() != int32(srv) {
		switch m.srv.Load() {
		case SRV_MODEL:
			if m.lastNumber.Load() > 0 {
				m.fs.Anchor(m.lastNumber.Load())
				m.fs.Flush()
				log.Debug("Model srv flush", "last", m.lastNumber.Load())
			}
		case SRV_RECORD:
			if m.lastNumber.Load() > 0 {
				log.Debug("Record srv flush", "last", m.lastNumber.Load())
				m.engine.Set([]byte("srv_record_last"), []byte(strconv.FormatUint(m.lastNumber.Load(), 16)))
			}
		default:
			return errors.New("Invalid current service")
		}

		switch srv {
		case SRV_MODEL:
			m.fs.InitBlockNumber()
			m.lastNumber.Store(m.fs.LastListenBlockNumber())
			log.Debug("Model srv load", "last", m.lastNumber.Load())
		case SRV_RECORD:
			if v := m.engine.Get([]byte("srv_record_last")); v != nil {
				if number, err := strconv.ParseUint(string(v), 16, 64); err == nil {
					m.lastNumber.Store(number)
				} else {
					m.lastNumber.Store(0)
				}
			} else {
				m.lastNumber.Store(0)
			}
			log.Debug("Record srv load", "last", m.lastNumber.Load())
		default:
			return errors.New("Unknow service")
		}
		m.srv.Store(int32(srv))
		log.Info("Service switch", "srv", m.srv.Load(), "last", m.lastNumber.Load())
	}

	return nil
}

// only for examples
func (m *Monitor) forExplorerService(block *types.Block) error {
	return errors.New("not support")
}

func (m *Monitor) forExchangeService(block *types.Block) error {
	return errors.New("not support")
}

func (m *Monitor) forRecordService(block *types.Block) error {
	if block.Number%4096 == 0 {
		log.Info("Block record", "num", block.Number, "hash", block.Hash, "txs", len(block.Txs), "last", m.lastNumber.Load())
	}
	if len(block.Txs) > 0 {
		for _, t := range block.Txs {
			x := new(big.Float).Quo(new(big.Float).SetInt(t.Amount), new(big.Float).SetInt(big.NewInt(params1.Cortex)))
			log.Debug("Tx record", "hash", t.Hash, "amount", x, "gas", t.GasLimit, "receipt", t.Recipient, "payload", t.Payload)

			if v, err := json.Marshal(t); err != nil {
				return err
			} else {
				m.engine.Set(t.Hash.Bytes(), v)
			}
		}
	}

	if v, err := json.Marshal(block); err != nil {
		return err
	} else {
		m.engine.Set(block.Hash.Bytes(), v)
	}

	m.engine.Set([]byte("srv_record_last"), []byte(strconv.FormatUint(block.Number, 16)))
	return nil
}

func (m *Monitor) forModelService(block *types.Block) error {
	i := block.Number
	if i%65536 == 0 {
		defer func() {
			//elapsedA := time.Duration(mclock.Now()) - time.Duration(m.start)
			//log.Info("Nas monitor", "start", m.startNumber.Load(), "max", uint64(m.CurrentNumber()), "last", m.lastNumber.Load(), "cur", i, "bps", math.Abs(float64(i)-float64(m.startNumber.Load()))*1000*1000*1000/float64(elapsedA), "elapsed", common.PrettyDuration(elapsedA), "scope", m.scope, "db", common.PrettyDuration(m.fs.Metrics()), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "root", m.fs.Root())
			m.fs.SkipPrint()
		}()
	}

	if hash, suc := m.blockCache.Get(i); !suc || hash != block.Hash.Hex() {
		if record, parseErr := m.parseBlockTorrentInfo(block); parseErr != nil {
			log.Error("Parse new block", "number", block.Number, "block", block, "error", parseErr)
			return parseErr
		} else if record {
			elapsed := time.Duration(mclock.Now()) - time.Duration(m.start)

			if m.ckp != nil && m.ckp.TfsCheckPoint > 0 && i == m.ckp.TfsCheckPoint {
				if common.BytesToHash(m.fs.GetRoot(i)) == m.ckp.TfsRoot {
					log.Warn("FIRST MILESTONE PASS", "number", i, "root", m.fs.Root(), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "elapsed", common.PrettyDuration(elapsed))
				} else {
					log.Error("Fs checkpoint failed", "number", i, "root", m.fs.Root(), "blocks", len(m.fs.Blocks()), "files", len(m.fs.Files()), "txs", m.fs.Txs(), "elapsed", common.PrettyDuration(elapsed), "exp", m.ckp.TfsRoot, "leaves", len(m.fs.Leaves()))
					panic("FIRST MILESTONE ERROR, run './cortex removedb' command to solve this problem")
				}
			}

			log.Debug("Seal fs record", "number", i, "record", record, "root", m.fs.Root().Hex(), "blocks", len(m.fs.Blocks()), "txs", m.fs.Txs(), "files", len(m.fs.Files()), "ckp", m.fs.CheckPoint())
		} else {
			if m.fs.LastListenBlockNumber() < i {
				m.fs.Anchor(i)
			}

			log.Trace("Confirm to seal the fs record", "number", i)
		}
		m.blockCache.Add(i, block.Hash.Hex())
	}

	return nil
}
