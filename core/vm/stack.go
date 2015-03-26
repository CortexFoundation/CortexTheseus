package vm

import (
	"fmt"
	"math/big"
)

const maxStack = 1024

func newStack() *stack {
	return &stack{}
}

type stack struct {
	data []*big.Int
	ptr  int
}

func (st *stack) push(d *big.Int) {
	if len(st.data) == maxStack {
		panic(fmt.Sprintf("stack limit reached (%d)", maxStack))
	}

	stackItem := new(big.Int).Set(d)
	if len(st.data) > st.ptr {
		st.data[st.ptr] = stackItem
	} else {
		st.data = append(st.data, stackItem)
	}
	st.ptr++
}

func (st *stack) pop() (ret *big.Int) {
	st.ptr--
	ret = st.data[st.ptr]
	return
}

func (st *stack) len() int {
	return st.ptr
}

func (st *stack) swap(n int) {
	st.data[st.len()-n], st.data[st.len()-1] = st.data[st.len()-1], st.data[st.len()-n]
}

func (st *stack) dup(n int) {
	st.push(st.data[st.len()-n])
}

func (st *stack) peek() *big.Int {
	return st.data[st.len()-1]
}

func (st *stack) require(n int) {
	if st.len() < n {
		panic(fmt.Sprintf("stack underflow (%d <=> %d)", len(st.data), n))
	}
}

func (st *stack) Print() {
	fmt.Println("### stack ###")
	if len(st.data) > 0 {
		for i, val := range st.data {
			fmt.Printf("%-3d  %v\n", i, val)
		}
	} else {
		fmt.Println("-- empty --")
	}
	fmt.Println("#############")
}
