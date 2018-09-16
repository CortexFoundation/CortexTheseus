package vm

import (
	"fmt"
	"testing"
)

func TestLocalInfer(t *testing.T) {
	modelDir := "~/CortexNet/infer_data/model"
	inputDir := "~/CortexNet/infer_data/image"
	label, err := LocalInfer(modelDir, inputDir)

	fmt.Println(label, err)
}
