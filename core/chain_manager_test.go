package core

import (
	"bytes"
	"fmt"
	"math/big"
	"os"
	"path"
	"runtime"
	"strconv"
	"testing"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethdb"
	"github.com/ethereum/go-ethereum/event"
	"github.com/ethereum/go-ethereum/rlp"
)

func init() {
	runtime.GOMAXPROCS(runtime.NumCPU())
}

// Test fork of length N starting from block i
func testFork(t *testing.T, bman *BlockProcessor, i, N int, f func(td1, td2 *big.Int)) {
	// switch databases to process the new chain
	db, err := ethdb.NewMemDatabase()
	if err != nil {
		t.Fatal("Failed to create db:", err)
	}
	// copy old chain up to i into new db with deterministic canonical
	bman2, err := newCanonical(i, db)
	if err != nil {
		t.Fatal("could not make new canonical in testFork", err)
	}
	// asert the bmans have the same block at i
	bi1 := bman.bc.GetBlockByNumber(uint64(i)).Hash()
	bi2 := bman2.bc.GetBlockByNumber(uint64(i)).Hash()
	if bytes.Compare(bi1, bi2) != 0 {
		t.Fatal("chains do not have the same hash at height", i)
	}

	bman2.bc.SetProcessor(bman2)

	// extend the fork
	parent := bman2.bc.CurrentBlock()
	chainB := makeChain(bman2, parent, N, db, ForkSeed)
	err = bman2.bc.InsertChain(chainB)
	if err != nil {
		t.Fatal("Insert chain error for fork:", err)
	}

	tdpre := bman.bc.Td()
	// Test the fork's blocks on the original chain
	td, err := testChain(chainB, bman)
	if err != nil {
		t.Fatal("expected chainB not to give errors:", err)
	}
	// Compare difficulties
	f(tdpre, td)
}

func printChain(bc *ChainManager) {
	for i := bc.CurrentBlock().Number().Uint64(); i > 0; i-- {
		b := bc.GetBlockByNumber(uint64(i))
		fmt.Printf("\t%x\n", b.Hash())
	}
}

// process blocks against a chain
func testChain(chainB types.Blocks, bman *BlockProcessor) (*big.Int, error) {
	td := new(big.Int)
	for _, block := range chainB {
		td2, err := bman.bc.processor.Process(block)
		if err != nil {
			if IsKnownBlockErr(err) {
				continue
			}
			return nil, err
		}
		block.Td = td2
		td = td2

		bman.bc.mu.Lock()
		{
			bman.bc.write(block)
		}
		bman.bc.mu.Unlock()
	}
	return td, nil
}

func loadChain(fn string, t *testing.T) (types.Blocks, error) {
	fh, err := os.OpenFile(path.Join(os.Getenv("GOPATH"), "src", "github.com", "ethereum", "go-ethereum", "_data", fn), os.O_RDONLY, os.ModePerm)
	if err != nil {
		return nil, err
	}
	defer fh.Close()

	var chain types.Blocks
	if err := rlp.Decode(fh, &chain); err != nil {
		return nil, err
	}

	return chain, nil
}

func insertChain(done chan bool, chainMan *ChainManager, chain types.Blocks, t *testing.T) {
	err := chainMan.InsertChain(chain)
	if err != nil {
		fmt.Println(err)
		t.FailNow()
	}
	done <- true
}

func TestExtendCanonical(t *testing.T) {
	CanonicalLength := 5
	db, err := ethdb.NewMemDatabase()
	if err != nil {
		t.Fatal("Failed to create db:", err)
	}
	// make first chain starting from genesis
	bman, err := newCanonical(CanonicalLength, db)
	if err != nil {
		t.Fatal("Could not make new canonical chain:", err)
	}
	f := func(td1, td2 *big.Int) {
		if td2.Cmp(td1) <= 0 {
			t.Error("expected chainB to have higher difficulty. Got", td2, "expected more than", td1)
		}
	}
	// Start fork from current height (CanonicalLength)
	testFork(t, bman, CanonicalLength, 1, f)
	testFork(t, bman, CanonicalLength, 2, f)
	testFork(t, bman, CanonicalLength, 5, f)
	testFork(t, bman, CanonicalLength, 10, f)
}

func TestShorterFork(t *testing.T) {
	db, err := ethdb.NewMemDatabase()
	if err != nil {
		t.Fatal("Failed to create db:", err)
	}
	// make first chain starting from genesis
	bman, err := newCanonical(10, db)
	if err != nil {
		t.Fatal("Could not make new canonical chain:", err)
	}
	f := func(td1, td2 *big.Int) {
		if td2.Cmp(td1) >= 0 {
			t.Error("expected chainB to have lower difficulty. Got", td2, "expected less than", td1)
		}
	}
	// Sum of numbers must be less than 10
	// for this to be a shorter fork
	testFork(t, bman, 0, 3, f)
	testFork(t, bman, 0, 7, f)
	testFork(t, bman, 1, 1, f)
	testFork(t, bman, 1, 7, f)
	testFork(t, bman, 5, 3, f)
	testFork(t, bman, 5, 4, f)
}

func TestLongerFork(t *testing.T) {
	db, err := ethdb.NewMemDatabase()
	if err != nil {
		t.Fatal("Failed to create db:", err)
	}
	// make first chain starting from genesis
	bman, err := newCanonical(10, db)
	if err != nil {
		t.Fatal("Could not make new canonical chain:", err)
	}
	f := func(td1, td2 *big.Int) {
		if td2.Cmp(td1) <= 0 {
			t.Error("expected chainB to have higher difficulty. Got", td2, "expected more than", td1)
		}
	}
	// Sum of numbers must be greater than 10
	// for this to be a longer fork
	testFork(t, bman, 0, 11, f)
	testFork(t, bman, 0, 15, f)
	testFork(t, bman, 1, 10, f)
	testFork(t, bman, 1, 12, f)
	testFork(t, bman, 5, 6, f)
	testFork(t, bman, 5, 8, f)
}

func TestEqualFork(t *testing.T) {
	db, err := ethdb.NewMemDatabase()
	if err != nil {
		t.Fatal("Failed to create db:", err)
	}
	bman, err := newCanonical(10, db)
	if err != nil {
		t.Fatal("Could not make new canonical chain:", err)
	}
	f := func(td1, td2 *big.Int) {
		if td2.Cmp(td1) != 0 {
			t.Error("expected chainB to have equal difficulty. Got", td2, "expected ", td1)
		}
	}
	// Sum of numbers must be equal to 10
	// for this to be an equal fork
	testFork(t, bman, 0, 10, f)
	testFork(t, bman, 1, 9, f)
	testFork(t, bman, 2, 8, f)
	testFork(t, bman, 5, 5, f)
	testFork(t, bman, 6, 4, f)
	testFork(t, bman, 9, 1, f)
}

func TestBrokenChain(t *testing.T) {
	db, err := ethdb.NewMemDatabase()
	if err != nil {
		t.Fatal("Failed to create db:", err)
	}
	bman, err := newCanonical(10, db)
	if err != nil {
		t.Fatal("Could not make new canonical chain:", err)
	}
	db2, err := ethdb.NewMemDatabase()
	if err != nil {
		t.Fatal("Failed to create db:", err)
	}
	bman2, err := newCanonical(10, db2)
	if err != nil {
		t.Fatal("Could not make new canonical chain:", err)
	}
	bman2.bc.SetProcessor(bman2)
	parent := bman2.bc.CurrentBlock()
	chainB := makeChain(bman2, parent, 5, db2, ForkSeed)
	chainB = chainB[1:]
	_, err = testChain(chainB, bman)
	if err == nil {
		t.Error("expected broken chain to return error")
	}
}

func TestChainInsertions(t *testing.T) {
	t.Skip() // travil fails.

	db, _ := ethdb.NewMemDatabase()

	chain1, err := loadChain("valid1", t)
	if err != nil {
		fmt.Println(err)
		t.FailNow()
	}

	chain2, err := loadChain("valid2", t)
	if err != nil {
		fmt.Println(err)
		t.FailNow()
	}

	var eventMux event.TypeMux
	chainMan := NewChainManager(db, db, &eventMux)
	txPool := NewTxPool(&eventMux)
	blockMan := NewBlockProcessor(db, db, nil, txPool, chainMan, &eventMux)
	chainMan.SetProcessor(blockMan)

	const max = 2
	done := make(chan bool, max)

	go insertChain(done, chainMan, chain1, t)
	go insertChain(done, chainMan, chain2, t)

	for i := 0; i < max; i++ {
		<-done
	}

	if bytes.Equal(chain2[len(chain2)-1].Hash(), chainMan.CurrentBlock().Hash()) {
		t.Error("chain2 is canonical and shouldn't be")
	}

	if !bytes.Equal(chain1[len(chain1)-1].Hash(), chainMan.CurrentBlock().Hash()) {
		t.Error("chain1 isn't canonical and should be")
	}
}

func TestChainMultipleInsertions(t *testing.T) {
	t.Skip() // travil fails.

	db, _ := ethdb.NewMemDatabase()

	const max = 4
	chains := make([]types.Blocks, max)
	var longest int
	for i := 0; i < max; i++ {
		var err error
		name := "valid" + strconv.Itoa(i+1)
		chains[i], err = loadChain(name, t)
		if len(chains[i]) >= len(chains[longest]) {
			longest = i
		}
		fmt.Println("loaded", name, "with a length of", len(chains[i]))
		if err != nil {
			fmt.Println(err)
			t.FailNow()
		}
	}
	var eventMux event.TypeMux
	chainMan := NewChainManager(db, db, &eventMux)
	txPool := NewTxPool(&eventMux)
	blockMan := NewBlockProcessor(db, db, nil, txPool, chainMan, &eventMux)
	chainMan.SetProcessor(blockMan)
	done := make(chan bool, max)
	for i, chain := range chains {
		// XXX the go routine would otherwise reference the same (chain[3]) variable and fail
		i := i
		chain := chain
		go func() {
			insertChain(done, chainMan, chain, t)
			fmt.Println(i, "done")
		}()
	}

	for i := 0; i < max; i++ {
		<-done
	}

	if !bytes.Equal(chains[longest][len(chains[longest])-1].Hash(), chainMan.CurrentBlock().Hash()) {
		t.Error("Invalid canonical chain")
	}
}

func TestGetAncestors(t *testing.T) {
	t.Skip() // travil fails.

	db, _ := ethdb.NewMemDatabase()
	var eventMux event.TypeMux
	chainMan := NewChainManager(db, db, &eventMux)
	chain, err := loadChain("valid1", t)
	if err != nil {
		fmt.Println(err)
		t.FailNow()
	}

	for _, block := range chain {
		chainMan.write(block)
	}

	ancestors := chainMan.GetAncestors(chain[len(chain)-1], 4)
	fmt.Println(ancestors)
}
