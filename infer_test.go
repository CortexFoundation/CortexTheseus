package infernet

import (
	"fmt"
	"testing"
)

func TestLocalInfer(t *testing.T) {
	label, err := InferCore("./infer_data/model", "./infer_data/num_nine")

	fmt.Println(label, err)
}
