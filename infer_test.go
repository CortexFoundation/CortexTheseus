package infernet

import (
	"fmt"
	"testing"
)

func TestLocalInfer(t *testing.T) {
	label, err := InferCore("./infer_data/model", "./infer_data/num")

	fmt.Println(label, err)
}
