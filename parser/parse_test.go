package parser

import (
	"testing"
)

func TestParseModel(t *testing.T) {
	dir := "./infernet/infer_data/model/data/"
	dir = "../infer_data/model/data/"
	r := CheckModel(dir+"symbol", dir+"params")
	if r == nil {
		t.Log("model check ok")
	} else {
		t.Errorf("model check error: %v", r)
	}
}
