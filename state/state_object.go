package state

import (
	"fmt"
	"math/big"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/trie"
)

type Code []byte

func (self Code) String() string {
	return string(self) //strings.Join(Disassemble(self), " ")
}

type Storage map[string]*ethutil.Value

func (self Storage) Copy() Storage {
	cpy := make(Storage)
	for key, value := range self {
		// XXX Do we need a 'value' copy or is this sufficient?
		cpy[key] = value
	}

	return cpy
}

type StateObject struct {
	db ethutil.Database
	// Address of the object
	address []byte
	// Shared attributes
	balance  *big.Int
	codeHash []byte
	Nonce    uint64
	// Contract related attributes
	State    *StateDB
	Code     Code
	InitCode Code

	storage Storage

	// Total gas pool is the total amount of gas currently
	// left if this object is the coinbase. Gas is directly
	// purchased of the coinbase.
	gasPool *big.Int

	// Mark for deletion
	// When an object is marked for deletion it will be delete from the trie
	// during the "update" phase of the state transition
	remove bool
}

func (self *StateObject) Reset() {
	self.storage = make(Storage)
	self.State.Reset()
}

func NewStateObject(addr []byte, db ethutil.Database) *StateObject {
	// This to ensure that it has 20 bytes (and not 0 bytes), thus left or right pad doesn't matter.
	address := ethutil.Address(addr)

	object := &StateObject{db: db, address: address, balance: new(big.Int), gasPool: new(big.Int)}
	object.State = New(nil, db) //New(trie.New(ethutil.Config.Db, ""))
	object.storage = make(Storage)
	object.gasPool = new(big.Int)

	return object
}

func NewStateObjectFromBytes(address, data []byte, db ethutil.Database) *StateObject {
	object := &StateObject{address: address, db: db}
	object.RlpDecode(data)

	return object
}

func (self *StateObject) MarkForDeletion() {
	self.remove = true
	statelogger.DebugDetailf("%x: #%d %v (deletion)\n", self.Address(), self.Nonce, self.balance)
}

func (c *StateObject) getAddr(addr []byte) *ethutil.Value {
	return ethutil.NewValueFromBytes([]byte(c.State.trie.Get(addr)))
}

func (c *StateObject) setAddr(addr []byte, value interface{}) {
	c.State.trie.Update(addr, ethutil.Encode(value))
}

func (self *StateObject) GetStorage(key *big.Int) *ethutil.Value {
	return self.GetState(key.Bytes())
}
func (self *StateObject) SetStorage(key *big.Int, value *ethutil.Value) {
	self.SetState(key.Bytes(), value)
}

func (self *StateObject) Storage() map[string]*ethutil.Value {
	return self.storage
}

func (self *StateObject) GetState(k []byte) *ethutil.Value {
	key := ethutil.LeftPadBytes(k, 32)

	value := self.storage[string(key)]
	if value == nil {
		value = self.getAddr(key)

		if !value.IsNil() {
			self.storage[string(key)] = value
		}
	}

	return value
}

func (self *StateObject) SetState(k []byte, value *ethutil.Value) {
	key := ethutil.LeftPadBytes(k, 32)
	self.storage[string(key)] = value.Copy()
}

func (self *StateObject) Sync() {
	for key, value := range self.storage {
		if value.Len() == 0 {
			self.State.trie.Delete([]byte(key))
			continue
		}

		self.setAddr([]byte(key), value)
	}
}

func (c *StateObject) GetInstr(pc *big.Int) *ethutil.Value {
	if int64(len(c.Code)-1) < pc.Int64() {
		return ethutil.NewValue(0)
	}

	return ethutil.NewValueFromBytes([]byte{c.Code[pc.Int64()]})
}

func (c *StateObject) AddBalance(amount *big.Int) {
	c.SetBalance(new(big.Int).Add(c.balance, amount))

	statelogger.Debugf("%x: #%d %v (+ %v)\n", c.Address(), c.Nonce, c.balance, amount)
}
func (c *StateObject) AddAmount(amount *big.Int) { c.AddBalance(amount) }

func (c *StateObject) SubBalance(amount *big.Int) {
	c.SetBalance(new(big.Int).Sub(c.balance, amount))

	statelogger.Debugf("%x: #%d %v (- %v)\n", c.Address(), c.Nonce, c.balance, amount)
}
func (c *StateObject) SubAmount(amount *big.Int) { c.SubBalance(amount) }

func (c *StateObject) SetBalance(amount *big.Int) {
	c.balance = amount
}

func (self *StateObject) Balance() *big.Int { return self.balance }

//
// Gas setters and getters
//

// Return the gas back to the origin. Used by the Virtual machine or Closures
func (c *StateObject) ReturnGas(gas, price *big.Int) {}
func (c *StateObject) ConvertGas(gas, price *big.Int) error {
	total := new(big.Int).Mul(gas, price)
	if total.Cmp(c.balance) > 0 {
		return fmt.Errorf("insufficient amount: %v, %v", c.balance, total)
	}

	c.SubAmount(total)

	return nil
}

func (self *StateObject) SetGasPool(gasLimit *big.Int) {
	self.gasPool = new(big.Int).Set(gasLimit)

	statelogger.Debugf("%x: gas (+ %v)", self.Address(), self.gasPool)
}

func (self *StateObject) BuyGas(gas, price *big.Int) error {
	if self.gasPool.Cmp(gas) < 0 {
		return GasLimitError(self.gasPool, gas)
	}

	rGas := new(big.Int).Set(gas)
	rGas.Mul(rGas, price)

	self.AddAmount(rGas)

	return nil
}

func (self *StateObject) RefundGas(gas, price *big.Int) {
	self.gasPool.Add(self.gasPool, gas)

	rGas := new(big.Int).Set(gas)
	rGas.Mul(rGas, price)

	self.balance.Sub(self.balance, rGas)
}

func (self *StateObject) Copy() *StateObject {
	stateObject := NewStateObject(self.Address(), self.db)
	stateObject.balance.Set(self.balance)
	stateObject.codeHash = ethutil.CopyBytes(self.codeHash)
	stateObject.Nonce = self.Nonce
	if self.State != nil {
		stateObject.State = self.State.Copy()
	}
	stateObject.Code = ethutil.CopyBytes(self.Code)
	stateObject.InitCode = ethutil.CopyBytes(self.InitCode)
	stateObject.storage = self.storage.Copy()
	stateObject.gasPool.Set(self.gasPool)
	stateObject.remove = self.remove

	return stateObject
}

func (self *StateObject) Set(stateObject *StateObject) {
	*self = *stateObject
}

//
// Attribute accessors
//

func (c *StateObject) N() *big.Int {
	return big.NewInt(int64(c.Nonce))
}

// Returns the address of the contract/account
func (c *StateObject) Address() []byte {
	return c.address
}

// Returns the initialization Code
func (c *StateObject) Init() Code {
	return c.InitCode
}

func (self *StateObject) Trie() *trie.Trie {
	return self.State.trie
}

func (self *StateObject) Root() []byte {
	return self.Trie().Root()
}

func (self *StateObject) SetCode(code []byte) {
	self.Code = code
}

//
// Encoding
//

// State object encoding methods
func (c *StateObject) RlpEncode() []byte {
	return ethutil.Encode([]interface{}{c.Nonce, c.balance, c.Root(), c.CodeHash()})
}

func (c *StateObject) CodeHash() ethutil.Bytes {
	return crypto.Sha3(c.Code)
}

func (c *StateObject) RlpDecode(data []byte) {
	decoder := ethutil.NewValueFromBytes(data)

	c.Nonce = decoder.Get(0).Uint()
	c.balance = decoder.Get(1).BigInt()
	c.State = New(decoder.Get(2).Bytes(), c.db) //New(trie.New(ethutil.Config.Db, decoder.Get(2).Interface()))
	c.storage = make(map[string]*ethutil.Value)
	c.gasPool = new(big.Int)

	c.codeHash = decoder.Get(3).Bytes()

	c.Code, _ = c.db.Get(c.codeHash)
}

// Storage change object. Used by the manifest for notifying changes to
// the sub channels.
type StorageState struct {
	StateAddress []byte
	Address      []byte
	Value        *big.Int
}
