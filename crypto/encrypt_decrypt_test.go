package crypto

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/ethereum/go-ethereum/ethutil"
)

func TestBox(t *testing.T) {
	prv1 := ToECDSA(ethutil.Hex2Bytes("4b50fa71f5c3eeb8fdc452224b2395af2fcc3d125e06c32c82e048c0559db03f"))
	prv2 := ToECDSA(ethutil.Hex2Bytes("d0b043b4c5d657670778242d82d68a29d25d7d711127d17b8e299f156dad361a"))
	pub2 := ToECDSAPub(ethutil.Hex2Bytes("04bd27a63c91fe3233c5777e6d3d7b39204d398c8f92655947eb5a373d46e1688f022a1632d264725cbc7dc43ee1cfebde42fa0a86d08b55d2acfbb5e9b3b48dc5"))

	message := []byte("Hello, world.")
	ct, err := Encrypt(pub2, message)
	if err != nil {
		fmt.Println(err.Error())
		t.FailNow()
	}

	pt, err := Decrypt(prv2, ct)
	if err != nil {
		fmt.Println(err.Error())
		t.FailNow()
	}

	if !bytes.Equal(pt, message) {
		fmt.Println("ecies: plaintext doesn't match message")
		t.FailNow()
	}

	_, err = Decrypt(prv1, pt)
	if err == nil {
		fmt.Println("ecies: encryption should not have succeeded")
		t.FailNow()
	}

}
