package secp256k1

import (
	"fmt"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/crypto/secp256k1"
	dcred_secp256k1 "github.com/decred/dcrd/dcrec/secp256k1/v4"
)

func TestFuzzer(t *testing.T) {
	a, b := "00000000N0000000/R0000000000000000", "0U0000S0000000mkhP000000000000000U"
	fuzz([]byte(a), []byte(b))
}

func Fuzz(f *testing.F) {
	f.Fuzz(func(t *testing.T, a, b []byte) {
		fuzz(a, b)
	})
}

func fuzz(dataP1, dataP2 []byte) {
	var (
		curveA = secp256k1.S256()
		curveB = dcred_secp256k1.S256()
	)
	// first point
	x1, y1 := curveB.ScalarBaseMult(dataP1)
	// second points
	x2, y2 := curveB.ScalarBaseMult(dataP2)
	resAX, resAY := curveA.Add(x1, y1, x2, y2)
	resBX, resBY := curveB.Add(x1, y1, x2, y2)
	if resAX.Cmp(resBX) != 0 || resAY.Cmp(resBY) != 0 {
		fmt.Printf("%s %s %s %s\n", x1, y1, x2, y2)
		panic(fmt.Sprintf("Addition failed: geth: %s %s btcd: %s %s", resAX, resAY, resBX, resBY))
	}
}
