package vm

import (
	"fmt"
	"testing"
)

func TestLocalInfer(t *testing.T) {
	modelDir := "/home/wlt/CortexNet/infer_data/model"
	inputDir := "/home/wlt/CortexNet/infer_data/image"
	label, err := LocalInfer(modelDir, inputDir)

	fmt.Println(label, err)
}
