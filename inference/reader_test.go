package inference

//go:generate go run gen.go reader_test.template

import (
	"io/ioutil"
	"strings"
)

func getFlist(dtype string) []string {
	infolist, err := ioutil.ReadDir("data")
	if err != nil {
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
