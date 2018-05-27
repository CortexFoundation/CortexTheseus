package types

import (
	"fmt"
	"testing"
)

var errorCode = []byte{0x0}
var modelCode = []byte{0x0, 0x1, 0x2}
var inputCode = []byte{0x0, 0x2, 0x3}

func testGas(im *ModelMeta) {
	fmt.Println(im.Gas())
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
			if res.RawSize() != testObj.need {
				t.Errorf("test %d, length should be %d but get %d", i, testObj.need, res.RawSize())
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
			if res.RawSize() != testObj.need {
				t.Errorf("test %d, length should be %d but get %d", i, testObj.need, res.RawSize())
			}
			// testGas(res)
		}
	}
}
