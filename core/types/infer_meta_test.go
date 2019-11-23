package types

import (
	"errors"
	"fmt"
	"testing"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	simplejson "github.com/bitly/go-simplejson"
)

var errorCode = []byte{0x0}
var modelCode = []byte{0x0, 0x1, 0x2}
var inputCode = []byte{0x0, 0x2, 0x3}

var ErrorCodeLengthNotEnough = errors.New("Code length is less than expected.")

func testGas(im *ModelMeta) {
	fmt.Println(im.Gas)
}

var testModelList = []struct {
	function func([]byte) (*ModelMeta, error)
	param    []byte
	need     uint64
	err      error
}{
	{
		function: ParseModelMeta,
		param:    errorCode,
		need:     0,
		err:      ErrorCodeLengthNotEnough,
	},
	{
		function: ParseModelMeta,
		param:    modelCode,
		need:     1,
		err:      nil,
	},
	{
		function: ParseModelMeta,
		param:    inputCode,
		need:     0,
		err:      ErrorCodeTypeModelMeta,
	},
}
var testInputList = []struct {
	function func([]byte) (*InputMeta, error)
	param    []byte
	need     uint64
	err      error
}{
	{
		function: ParseInputMeta,
		param:    errorCode,
		need:     0,
		err:      ErrorCodeLengthNotEnough,
	},
	{
		function: ParseInputMeta,
		param:    modelCode,
		need:     0,
		err:      ErrorCodeTypeInputMeta,
	},
	{
		function: ParseInputMeta,
		param:    inputCode,
		need:     1,
		err:      nil,
	},
}

func TestMeta(t *testing.T) {

	for i, testObj := range testModelList {
		res, err := testObj.function(testObj.param)
		// fuck(res)
		if err != testObj.err {
			t.Errorf("test %d, error need %s but return %s", i, testObj.err, err)
		}
		if err == nil {
			if res.RawSize != testObj.need {
				t.Errorf("test %d, length should be %d but get %d", i, testObj.need, res.RawSize)
			}
			testGas(res)
		}
	}
	for i, testObj := range testInputList {
		res, err := testObj.function(testObj.param)
		// fmt.Println(res.Gas())
		if err != testObj.err {
			t.Errorf("test %d, error need %s but return %s", i, testObj.err, err)
		}
		if err == nil {
			if res.RawSize != testObj.need {
				t.Errorf("test %d, length should be %d but get %d", i, testObj.need, res.RawSize)
			}
			// testGas(res)
		}
	}
}

func TestShit(t *testing.T) {
	mh := common.HexToHash("0x5c4d1f84063be8e25e83da6452b1821926548b3c2a2a903a0724e14d5c917b00")
	ih := common.HexToHash("0xc0a1f3c82e11e314822679e4834e3bc575bd017d12d888acda4a851a62d261dc")
	testModelMeta := &ModelMeta{
		Hash:          mh,
		RawSize:       10000,
		InputShape:    []uint64{10, 1},
		OutputShape:   []uint64{1},
		Gas:           100000,
		AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x2, 0x2})),
	}
	// new a modelmeta at 0x1001 and new a datameta at 0x2001

	testInputMeta := &InputMeta{
		Hash:    ih,
		RawSize: 10000,
		Shape:   []uint64{1},
		//AuthorAddress: common.BytesToAddress(crypto.Keccak256([]byte{0x3})),
	}
	s := `
	{"info": "{\"Hash\": \"0x5c4d1f84063be8e25e83da6452b1821926548b3c2a2a903a0724e14d5c917b00\", \"AuthorAddress\": \"0x0553b0185a35cd5bb6386747517ef7e53b15e287\", \"RawSize\": 45401702, \"InputShape\": [3, 224, 224], \"OutputShape\": [1], \"Gas\": 45401702}", "msg": "ok"}
	`
	s1 := `
	{"info": "{\"Hash\": \"0xc0a1f3c82e11e314822679e4834e3bc575bd017d12d888acda4a851a62d261dc\", \"RawSize\": 150656, \"Shape\": [3, 224, 224]}", "msg": "ok"}
`
	js, _ := simplejson.NewJson([]byte(s))
	js1, _ := simplejson.NewJson([]byte(s1))
	ss, _ := js.Get("info").String()
	ss1, _ := js1.Get("info").String()
	testModelMeta.DecodeJSON(ss)
	testModelMeta.AuthorAddress = common.BytesToAddress([]byte{0x0})
	s, _ = testModelMeta.EncodeJSON()
	t.Errorf(s)
	testInputMeta.DecodeJSON(ss1)
	t.Errorf(testInputMeta.EncodeJSON())
	//t.Errorf(string(testInputMeta.AuthorAddress[:]))
}
