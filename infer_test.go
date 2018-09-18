package infernet

import (
	"fmt"
	"testing"
)

func TestLocalInfer(t *testing.T) {
	modelCfg := "./infer_data/model/data/params"
	modelBin := "./infer_data/model/data/symbol"
	image := "./infer_data/num_nine/data"
	label, err := InferCore(modelCfg, modelBin, image)
	// label, err := InferCore("./infer_data/model", "./infer_data/num")

	fmt.Println(label, err)
}
