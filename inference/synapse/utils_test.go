package synapse

import (
	"github.com/CortexFoundation/CortexTheseus/inference"
	"io/ioutil"
	"os"
	"path"
	"strings"
	"testing"
	//"fmt"
)

func getFlist(dtype string) []string {

	infolist, err := ioutil.ReadDir("../data")
	if err != nil || len(infolist) == 0 {
		panic(err)
	}
	files := make([]string, 0)
	for _, v := range infolist {
		f := v.Name()
		if strings.Contains(f, dtype) {
			files = append(files, f)
		}
	}

	return files
}

func TestRLPHashString(t *testing.T) {
	var rlp string
	files := getFlist("i1")
	if len(files) == 0 {
		t.Fatal("no files")
	}
	for _, fname := range files {

		fid, err := os.Open(path.Join("..", "data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := inference.NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetInt8()
		if err != nil {
			panic(err)
		}

		rlp = RLPHashString(data)
		t.Log("rlp hash", "data", data, "rlp", rlp)
	}
}
