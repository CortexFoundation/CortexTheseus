package ethpub

import (
	"encoding/json"
	"fmt"
	"github.com/ethereum/eth-go/ethchain"
	"github.com/ethereum/eth-go/ethcrypto"
	"github.com/ethereum/eth-go/ethtrie"
	"github.com/ethereum/eth-go/ethutil"
	"strings"
)

// Peer interface exposed to QML

type PPeer struct {
	ref          *ethchain.Peer
	Inbound      bool   `json:"isInbound"`
	LastSend     int64  `json:"lastSend"`
	LastPong     int64  `json:"lastPong"`
	Ip           string `json:"ip"`
	Port         int    `json:"port"`
	Version      string `json:"version"`
	LastResponse string `json:"lastResponse"`
	Latency      string `json:"latency"`
}

func NewPPeer(peer ethchain.Peer) *PPeer {
	if peer == nil {
		return nil
	}

	// TODO: There must be something build in to do this?
	var ip []string
	for _, i := range peer.Host() {
		ip = append(ip, fmt.Sprintf("%d", i))
	}
	ipAddress := strings.Join(ip, ".")

	return &PPeer{ref: &peer, Inbound: peer.Inbound(), LastSend: peer.LastSend().Unix(), LastPong: peer.LastPong(), Version: peer.Version(), Ip: ipAddress, Port: int(peer.Port()), Latency: peer.PingTime()}
}

// Block interface exposed to QML
type PBlock struct {
	ref          *ethchain.Block
	Number       int    `json:"number"`
	Hash         string `json:"hash"`
	Transactions string `json:"transactions"`
	Time         int64  `json:"time"`
	Coinbase     string `json:"coinbase"`
	Name         string `json:"name"`
	GasLimit     string `json:"gasLimit"`
	GasUsed      string `json:"gasUsed"`
}

// Creates a new QML Block from a chain block
func NewPBlock(block *ethchain.Block) *PBlock {
	if block == nil {
		return nil
	}

	var ptxs []PTx
	for _, tx := range block.Transactions() {
		ptxs = append(ptxs, *NewPTx(tx))
	}

	txJson, err := json.Marshal(ptxs)
	if err != nil {
		return nil
	}

	return &PBlock{ref: block, Number: int(block.Number.Uint64()), GasUsed: block.GasUsed.String(), GasLimit: block.GasLimit.String(), Hash: ethutil.Bytes2Hex(block.Hash()), Transactions: string(txJson), Time: block.Time, Coinbase: ethutil.Bytes2Hex(block.Coinbase)}
}

func (self *PBlock) ToString() string {
	if self.ref != nil {
		return self.ref.String()
	}

	return ""
}

func (self *PBlock) GetTransaction(hash string) *PTx {
	tx := self.ref.GetTransaction(ethutil.Hex2Bytes(hash))
	if tx == nil {
		return nil
	}

	return NewPTx(tx)
}

type PTx struct {
	ref *ethchain.Transaction

	Value           string `json:"value"`
	Gas             string `json:"gas"`
	GasPrice        string `json:"gasPrice"`
	Hash            string `json:"hash"`
	Address         string `json:"address"`
	Sender          string `json:"sender"`
	RawData         string `json:"rawData"`
	Data            string `json:"data"`
	Contract        bool   `json:"isContract"`
	CreatesContract bool   `json:"createsContract"`
	Confirmations   int    `json:"confirmations"`
}

func NewPTx(tx *ethchain.Transaction) *PTx {
	hash := ethutil.Bytes2Hex(tx.Hash())
	receiver := ethutil.Bytes2Hex(tx.Recipient)
	if receiver == "0000000000000000000000000000000000000000" {
		receiver = ethutil.Bytes2Hex(tx.CreationAddress())
	}
	sender := ethutil.Bytes2Hex(tx.Sender())
	createsContract := tx.CreatesContract()

	var data string
	if tx.CreatesContract() {
		data = strings.Join(ethchain.Disassemble(tx.Data), "\n")
	} else {
		data = ethutil.Bytes2Hex(tx.Data)
	}

	return &PTx{ref: tx, Hash: hash, Value: ethutil.CurrencyToString(tx.Value), Address: receiver, Contract: tx.CreatesContract(), Gas: tx.Gas.String(), GasPrice: tx.GasPrice.String(), Data: data, Sender: sender, CreatesContract: createsContract, RawData: ethutil.Bytes2Hex(tx.Data)}
}

func (self *PTx) ToString() string {
	return self.ref.String()
}

type PKey struct {
	Address    string `json:"address"`
	PrivateKey string `json:"privateKey"`
	PublicKey  string `json:"publicKey"`
}

func NewPKey(key *ethcrypto.KeyPair) *PKey {
	return &PKey{ethutil.Bytes2Hex(key.Address()), ethutil.Bytes2Hex(key.PrivateKey), ethutil.Bytes2Hex(key.PublicKey)}
}

type PReceipt struct {
	CreatedContract bool   `json:"createdContract"`
	Address         string `json:"address"`
	Hash            string `json:"hash"`
	Sender          string `json:"sender"`
}

func NewPReciept(contractCreation bool, creationAddress, hash, address []byte) *PReceipt {
	return &PReceipt{
		contractCreation,
		ethutil.Bytes2Hex(creationAddress),
		ethutil.Bytes2Hex(hash),
		ethutil.Bytes2Hex(address),
	}
}

type PStateObject struct {
	object *ethchain.StateObject
}

func NewPStateObject(object *ethchain.StateObject) *PStateObject {
	return &PStateObject{object: object}
}

func (c *PStateObject) GetStorage(address string) string {
	// Because somehow, even if you return nil to QML it
	// still has some magical object so we can't rely on
	// undefined or null at the QML side
	if c.object != nil {
		val := c.object.GetStorage(ethutil.Big("0x" + address))

		return val.BigInt().String()
	}

	return ""
}

func (c *PStateObject) Value() string {
	if c.object != nil {
		return c.object.Amount.String()
	}

	return ""
}

func (c *PStateObject) Address() string {
	if c.object != nil {
		return ethutil.Bytes2Hex(c.object.Address())
	}

	return ""
}

func (c *PStateObject) Nonce() int {
	if c.object != nil {
		return int(c.object.Nonce)
	}

	return 0
}

func (c *PStateObject) Root() string {
	if c.object != nil {
		return ethutil.Bytes2Hex(ethutil.NewValue(c.object.State().Root()).Bytes())
	}

	return "<err>"
}

func (c *PStateObject) IsContract() bool {
	if c.object != nil {
		return len(c.object.Script()) > 0
	}

	return false
}

func (self *PStateObject) EachStorage(cb ethtrie.EachCallback) {
	self.object.EachStorage(cb)
}

type KeyVal struct {
	Key   string
	Value string
}

func (c *PStateObject) StateKeyVal(asJson bool) interface{} {
	var values []KeyVal
	if c.object != nil {
		c.object.EachStorage(func(name string, value *ethutil.Value) {
			values = append(values, KeyVal{name, ethutil.Bytes2Hex(value.Bytes())})
		})
	}

	if asJson {
		valuesJson, err := json.Marshal(values)
		if err != nil {
			return nil
		}
		fmt.Println(string(valuesJson))
		return string(valuesJson)
	}

	return values
}

func (c *PStateObject) Script() string {
	if c.object != nil {
		return strings.Join(ethchain.Disassemble(c.object.Script()), " ")
	}

	return ""
}

func (c *PStateObject) HexScript() string {
	if c.object != nil {
		return ethutil.Bytes2Hex(c.object.Script())
	}

	return ""
}

type PStorageState struct {
	StateAddress string
	Address      string
	Value        string
}

func NewPStorageState(storageObject *ethchain.StorageState) *PStorageState {
	return &PStorageState{ethutil.Bytes2Hex(storageObject.StateAddress), ethutil.Bytes2Hex(storageObject.Address), storageObject.Value.String()}
}
