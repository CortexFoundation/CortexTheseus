package common

import (
	"fmt"
	"math/big"
	"time"
)

// Common big integers often used
var (
	Big1     = big.NewInt(1)
	Big2     = big.NewInt(2)
	Big3     = big.NewInt(3)
	Big0     = big.NewInt(0)
	BigTrue  = Big1
	BigFalse = Big0
	Big32    = big.NewInt(32)
	Big256   = big.NewInt(0xff)
	Big257   = big.NewInt(257)
)

func Bench(pre string, cb func()) {
	start := time.Now()
	cb()
	fmt.Println(pre, ": took:", time.Since(start))
}
