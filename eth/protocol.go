package eth

import (
	"bytes"
	"fmt"
	"math/big"

	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/errs"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/p2p"
	"github.com/ethereum/go-ethereum/rlp"
)

const (
	ProtocolVersion    = 58
	NetworkId          = 0
	ProtocolLength     = uint64(8)
	ProtocolMaxMsgSize = 10 * 1024 * 1024
	maxHashes          = 256
	maxBlocks          = 64
)

// eth protocol message codes
const (
	StatusMsg = iota
	GetTxMsg  // unused
	TxMsg
	GetBlockHashesMsg
	BlockHashesMsg
	GetBlocksMsg
	BlocksMsg
	NewBlockMsg
)

const (
	ErrMsgTooLarge = iota
	ErrDecode
	ErrInvalidMsgCode
	ErrProtocolVersionMismatch
	ErrNetworkIdMismatch
	ErrGenesisBlockMismatch
	ErrNoStatusMsg
	ErrExtraStatusMsg
)

var errorToString = map[int]string{
	ErrMsgTooLarge:             "Message too long",
	ErrDecode:                  "Invalid message",
	ErrInvalidMsgCode:          "Invalid message code",
	ErrProtocolVersionMismatch: "Protocol version mismatch",
	ErrNetworkIdMismatch:       "NetworkId mismatch",
	ErrGenesisBlockMismatch:    "Genesis block mismatch",
	ErrNoStatusMsg:             "No status message",
	ErrExtraStatusMsg:          "Extra status message",
}

// ethProtocol represents the ethereum wire protocol
// instance is running on each peer
type ethProtocol struct {
	txPool          txPool
	chainManager    chainManager
	blockPool       blockPool
	peer            *p2p.Peer
	id              string
	rw              p2p.MsgReadWriter
	errors          *errs.Errors
	protocolVersion int
	networkId       int
}

// backend is the interface the ethereum protocol backend should implement
// used as an argument to EthProtocol
type txPool interface {
	AddTransactions([]*types.Transaction)
	GetTransactions() types.Transactions
}

type chainManager interface {
	GetBlockHashesFromHash(hash []byte, amount uint64) (hashes [][]byte)
	GetBlock(hash []byte) (block *types.Block)
	Status() (td *big.Int, currentBlock []byte, genesisBlock []byte)
}

type blockPool interface {
	AddBlockHashes(next func() ([]byte, bool), peerId string)
	AddBlock(block *types.Block, peerId string)
	AddPeer(td *big.Int, currentBlock []byte, peerId string, requestHashes func([]byte) error, requestBlocks func([][]byte) error, peerError func(*errs.Error)) (best bool)
	RemovePeer(peerId string)
}

// message structs used for rlp decoding
type newBlockMsgData struct {
	Block *types.Block
	TD    *big.Int
}

type getBlockHashesMsgData struct {
	Hash   []byte
	Amount uint64
}

// main entrypoint, wrappers starting a server running the eth protocol
// use this constructor to attach the protocol ("class") to server caps
// the Dev p2p layer then runs the protocol instance on each peer
func EthProtocol(protocolVersion, networkId int, txPool txPool, chainManager chainManager, blockPool blockPool) p2p.Protocol {
	return p2p.Protocol{
		Name:    "eth",
		Version: uint(protocolVersion),
		Length:  ProtocolLength,
		Run: func(peer *p2p.Peer, rw p2p.MsgReadWriter) error {
			return runEthProtocol(protocolVersion, networkId, txPool, chainManager, blockPool, peer, rw)
		},
	}
}

// the main loop that handles incoming messages
// note RemovePeer in the post-disconnect hook
func runEthProtocol(protocolVersion, networkId int, txPool txPool, chainManager chainManager, blockPool blockPool, peer *p2p.Peer, rw p2p.MsgReadWriter) (err error) {
	id := peer.ID()
	self := &ethProtocol{
		txPool:          txPool,
		chainManager:    chainManager,
		blockPool:       blockPool,
		rw:              rw,
		peer:            peer,
		protocolVersion: protocolVersion,
		networkId:       networkId,
		errors: &errs.Errors{
			Package: "ETH",
			Errors:  errorToString,
		},
		id: fmt.Sprintf("%x", id[:8]),
	}
	err = self.handleStatus()
	if err == nil {
		self.propagateTxs()
		for {
			err = self.handle()
			if err != nil {
				self.blockPool.RemovePeer(self.id)
				break
			}
		}
	}
	return
}

func (self *ethProtocol) handle() error {
	msg, err := self.rw.ReadMsg()
	if err != nil {
		return err
	}
	if msg.Size > ProtocolMaxMsgSize {
		return self.protoError(ErrMsgTooLarge, "%v > %v", msg.Size, ProtocolMaxMsgSize)
	}
	// make sure that the payload has been fully consumed
	defer msg.Discard()

	switch msg.Code {
	case GetTxMsg: // ignore
	case StatusMsg:
		return self.protoError(ErrExtraStatusMsg, "")

	case TxMsg:
		// TODO: rework using lazy RLP stream
		var txs []*types.Transaction
		if err := msg.Decode(&txs); err != nil {
			return self.protoError(ErrDecode, "msg %v: %v", msg, err)
		}
		for _, tx := range txs {
			jsonlogger.LogJson(&logger.EthTxReceived{
				TxHash:   common.Bytes2Hex(tx.Hash()),
				RemoteId: self.peer.ID().String(),
			})
		}
		self.txPool.AddTransactions(txs)

	case GetBlockHashesMsg:
		var request getBlockHashesMsgData
		if err := msg.Decode(&request); err != nil {
			return self.protoError(ErrDecode, "->msg %v: %v", msg, err)
		}

		if request.Amount > maxHashes {
			request.Amount = maxHashes
		}
		hashes := self.chainManager.GetBlockHashesFromHash(request.Hash, request.Amount)
		return p2p.EncodeMsg(self.rw, BlockHashesMsg, common.ByteSliceToInterface(hashes)...)

	case BlockHashesMsg:
		msgStream := rlp.NewStream(msg.Payload)
		if _, err := msgStream.List(); err != nil {
			return err
		}

		var i int
		iter := func() (hash []byte, ok bool) {
			hash, err := msgStream.Bytes()
			if err == rlp.EOL {
				return nil, false
			} else if err != nil {
				self.protoError(ErrDecode, "msg %v: after %v hashes : %v", msg, i, err)
				return nil, false
			}
			i++
			return hash, true
		}
		self.blockPool.AddBlockHashes(iter, self.id)

	case GetBlocksMsg:
		msgStream := rlp.NewStream(msg.Payload)
		if _, err := msgStream.List(); err != nil {
			return err
		}

		var blocks []interface{}
		var i int
		for {
			i++
			var hash []byte
			if err := msgStream.Decode(&hash); err != nil {
				if err == rlp.EOL {
					break
				} else {
					return self.protoError(ErrDecode, "msg %v: %v", msg, err)
				}
			}
			block := self.chainManager.GetBlock(hash)
			if block != nil {
				blocks = append(blocks, block)
			}
			if i == maxBlocks {
				break
			}
		}
		return p2p.EncodeMsg(self.rw, BlocksMsg, blocks...)

	case BlocksMsg:
		msgStream := rlp.NewStream(msg.Payload)
		if _, err := msgStream.List(); err != nil {
			return err
		}
		for {
			var block types.Block
			if err := msgStream.Decode(&block); err != nil {
				if err == rlp.EOL {
					break
				} else {
					return self.protoError(ErrDecode, "msg %v: %v", msg, err)
				}
			}
			self.blockPool.AddBlock(&block, self.id)
		}

	case NewBlockMsg:
		var request newBlockMsgData
		if err := msg.Decode(&request); err != nil {
			return self.protoError(ErrDecode, "%v: %v", msg, err)
		}
		hash := request.Block.Hash()
		_, chainHead, _ := self.chainManager.Status()

		jsonlogger.LogJson(&logger.EthChainReceivedNewBlock{
			BlockHash:     common.Bytes2Hex(hash),
			BlockNumber:   request.Block.Number(), // this surely must be zero
			ChainHeadHash: common.Bytes2Hex(chainHead),
			BlockPrevHash: common.Bytes2Hex(request.Block.ParentHash()),
			RemoteId:      self.peer.ID().String(),
		})
		// to simplify backend interface adding a new block
		// uses AddPeer followed by AddBlock only if peer is the best peer
		// (or selected as new best peer)
		if self.blockPool.AddPeer(request.TD, hash, self.id, self.requestBlockHashes, self.requestBlocks, self.protoErrorDisconnect) {
			self.blockPool.AddBlock(request.Block, self.id)
		}

	default:
		return self.protoError(ErrInvalidMsgCode, "%v", msg.Code)
	}
	return nil
}

type statusMsgData struct {
	ProtocolVersion uint32
	NetworkId       uint32
	TD              *big.Int
	CurrentBlock    []byte
	GenesisBlock    []byte
}

func (self *ethProtocol) statusMsg() p2p.Msg {
	td, currentBlock, genesisBlock := self.chainManager.Status()

	return p2p.NewMsg(StatusMsg,
		uint32(self.protocolVersion),
		uint32(self.networkId),
		td,
		currentBlock,
		genesisBlock,
	)
}

func (self *ethProtocol) handleStatus() error {
	// send precanned status message
	if err := self.rw.WriteMsg(self.statusMsg()); err != nil {
		return err
	}

	// read and handle remote status
	msg, err := self.rw.ReadMsg()
	if err != nil {
		return err
	}

	if msg.Code != StatusMsg {
		return self.protoError(ErrNoStatusMsg, "first msg has code %x (!= %x)", msg.Code, StatusMsg)
	}

	if msg.Size > ProtocolMaxMsgSize {
		return self.protoError(ErrMsgTooLarge, "%v > %v", msg.Size, ProtocolMaxMsgSize)
	}

	var status statusMsgData
	if err := msg.Decode(&status); err != nil {
		return self.protoError(ErrDecode, "msg %v: %v", msg, err)
	}

	_, _, genesisBlock := self.chainManager.Status()

	if !bytes.Equal(status.GenesisBlock, genesisBlock) {
		return self.protoError(ErrGenesisBlockMismatch, "%x (!= %x)", status.GenesisBlock, genesisBlock)
	}

	if int(status.NetworkId) != self.networkId {
		return self.protoError(ErrNetworkIdMismatch, "%d (!= %d)", status.NetworkId, self.networkId)
	}

	if int(status.ProtocolVersion) != self.protocolVersion {
		return self.protoError(ErrProtocolVersionMismatch, "%d (!= %d)", status.ProtocolVersion, self.protocolVersion)
	}

	self.peer.Infof("Peer is [eth] capable (%d/%d). TD=%v H=%x\n", status.ProtocolVersion, status.NetworkId, status.TD, status.CurrentBlock[:4])

	self.blockPool.AddPeer(status.TD, status.CurrentBlock, self.id, self.requestBlockHashes, self.requestBlocks, self.protoErrorDisconnect)

	return nil
}

func (self *ethProtocol) requestBlockHashes(from []byte) error {
	self.peer.Debugf("fetching hashes (%d) %x...\n", maxHashes, from[0:4])
	return p2p.EncodeMsg(self.rw, GetBlockHashesMsg, interface{}(from), uint64(maxHashes))
}

func (self *ethProtocol) requestBlocks(hashes [][]byte) error {
	self.peer.Debugf("fetching %v blocks", len(hashes))
	return p2p.EncodeMsg(self.rw, GetBlocksMsg, common.ByteSliceToInterface(hashes)...)
}

func (self *ethProtocol) protoError(code int, format string, params ...interface{}) (err *errs.Error) {
	err = self.errors.New(code, format, params...)
	err.Log(self.peer.Logger)
	return
}

func (self *ethProtocol) protoErrorDisconnect(err *errs.Error) {
	err.Log(self.peer.Logger)
	if err.Fatal() {
		self.peer.Disconnect(p2p.DiscSubprotocolError)
	}
}

func (self *ethProtocol) propagateTxs() {
	transactions := self.txPool.GetTransactions()
	iface := make([]interface{}, len(transactions))
	for i, transaction := range transactions {
		iface[i] = transaction
	}

	self.rw.WriteMsg(p2p.NewMsg(TxMsg, iface...))
}
