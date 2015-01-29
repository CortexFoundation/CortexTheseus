package core

import (
	"bytes"
	"math"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/state"
)

type AccountChange struct {
	Address, StateAddress []byte
}

type FilterOptions struct {
	Earliest int64
	Latest   int64

	Address []byte
	Topics  [][]byte

	Skip int
	Max  int
}

// Filtering interface
type Filter struct {
	eth      EthManager
	earliest int64
	latest   int64
	skip     int
	address  []byte
	max      int
	topics   [][]byte

	BlockCallback func(*types.Block)
	LogsCallback  func(state.Logs)
}

// Create a new filter which uses a bloom filter on blocks to figure out whether a particular block
// is interesting or not.
func NewFilter(eth EthManager) *Filter {
	return &Filter{eth: eth}
}

func (self *Filter) SetOptions(options FilterOptions) {
	self.earliest = options.Earliest
	self.latest = options.Latest
	self.skip = options.Skip
	self.max = options.Max
	self.address = options.Address
	self.topics = options.Topics

}

// Set the earliest and latest block for filtering.
// -1 = latest block (i.e., the current block)
// hash = particular hash from-to
func (self *Filter) SetEarliestBlock(earliest int64) {
	self.earliest = earliest
}

func (self *Filter) SetLatestBlock(latest int64) {
	self.latest = latest
}

func (self *Filter) SetAddress(addr []byte) {
	self.address = addr
}

func (self *Filter) SetTopics(topics [][]byte) {
	self.topics = topics
}

func (self *Filter) SetMax(max int) {
	self.max = max
}

func (self *Filter) SetSkip(skip int) {
	self.skip = skip
}

// Run filters logs with the current parameters set
func (self *Filter) Find() state.Logs {
	earliestBlock := self.eth.ChainManager().CurrentBlock()
	var earliestBlockNo uint64 = uint64(self.earliest)
	if self.earliest == -1 {
		earliestBlockNo = earliestBlock.NumberU64()
	}
	var latestBlockNo uint64 = uint64(self.latest)
	if self.latest == -1 {
		latestBlockNo = earliestBlock.NumberU64()
	}

	var (
		logs  state.Logs
		block = self.eth.ChainManager().GetBlockByNumber(latestBlockNo)
		quit  bool
	)
	for i := 0; !quit && block != nil; i++ {
		// Quit on latest
		switch {
		case block.NumberU64() == earliestBlockNo, block.NumberU64() == 0:
			quit = true
		case self.max <= len(logs):
			break
		}

		// Use bloom filtering to see if this block is interesting given the
		// current parameters
		if self.bloomFilter(block) {
			// Get the logs of the block
			logs, err := self.eth.BlockProcessor().GetLogs(block)
			if err != nil {
				chainlogger.Warnln("err: filter get logs ", err)

				break
			}

			logs = append(logs, self.FilterLogs(logs)...)
		}

		block = self.eth.ChainManager().GetBlock(block.ParentHash())
	}

	skip := int(math.Min(float64(len(logs)), float64(self.skip)))

	return logs[skip:]
}

func includes(addresses [][]byte, a []byte) (found bool) {
	for _, addr := range addresses {
		if bytes.Compare(addr, a) == 0 {
			return true
		}
	}

	return
}

func (self *Filter) FilterLogs(logs state.Logs) state.Logs {
	var ret state.Logs

	// Filter the logs for interesting stuff
	for _, log := range logs {
		if len(self.address) > 0 && !bytes.Equal(self.address, log.Address()) {
			continue
		}

		for _, topic := range self.topics {
			if !includes(log.Topics(), topic) {
				continue
			}
		}

		ret = append(ret, log)
	}

	return ret
}

func (self *Filter) bloomFilter(block *types.Block) bool {
	if len(self.address) > 0 && !types.BloomLookup(block.Bloom(), self.address) {
		return false
	}

	for _, topic := range self.topics {
		if !types.BloomLookup(block.Bloom(), topic) {
			return false
		}
	}

	return true
}
