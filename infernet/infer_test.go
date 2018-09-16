package infernet

import (
	"fmt"
	"testing"
)

func TestLocalInfer(t *testing.T) {
	label, err := inferCore("./infer_data/model", "./infer_data/image")

	fmt.Println(label, err)
}
