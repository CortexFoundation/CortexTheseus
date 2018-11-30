package kernel

import (
	"path"
	"testing"
	"time"

	"github.com/ethereum/go-ethereum/inference"
)

func ReadImage(inputFilePath string) ([]byte, error) {
	r, rerr := inference.NewFileReader(inputFilePath)
	if rerr != nil {
		return nil, rerr
	}

	data, derr := r.GetBytes()
	if derr != nil {
		return nil, derr
	}

	// Tmp Code
	// DumpToFile("tmp.dump", data)

	return data, nil
}
func GetFilePath(hash string) string {
	return path.Join("/home/wlt/.cortex/storage", hash, "data")
}

func TestInferTime(t *testing.T) {
	t.Log("Test Infer Time initilized")
	const (
		length = 8
		loop   = 2000
	)

	var (
		// modelHash     = "4d8bc8272b882f315c6a96449ad4568fac0e6038"
		modelHash = "ca3d0286d5758697cdef653c1375960a868ac08a"
		// inputHashList = []string{
		// 	"6f67238dda00c9d1b2048e6f846481a8a6a59a07",
		// }
		inputHashList = []string{
			"18af0aff299483903f38e9c80c1c73288143c689",
			"265613d54a190df83ac67c2827c8ef1a071fd6a4",
			"521b31b82f9a1144f45acd52cc55fdb2c150b756",
			"7d942da381f32180a616cb1ef3515f71f9422c4a",
			"8839d6579fd4fbb147dd8194c52cfd2fb8d41603",
			"c35dde5292458e91c6533d671a9cfcf55fc46026",
			"ce5249145d1c007c13a5a23c34aaf34cf63c4cc2",
			"ed68da2d3d55b1c163c80f15d0f6490c88da644e",
		}

		inputBufferList [length][]byte
		start           time.Time
	)

	// Load image
	for i, hash := range inputHashList {
		buffer, err := ReadImage(GetFilePath(hash))
		if err != nil {
			t.Fatalf("Read image error: %s\n", err)
			return
		}

		inputBufferList[i] = buffer
	}
	t.Log("Input image loaded done")

	// Start test
	var load_du = time.Duration(0)
	var predict_du = time.Duration(0)
	var free_du = time.Duration(0)
	for i := 0; i < loop; i++ {
		// Load model
		modelPath := GetFilePath(modelHash)
		start = time.Now()
		network, mErr := LoadModel(path.Join(modelPath, "symbol"), path.Join(modelPath, "params"))
		load_du += time.Since(start)
		if mErr != nil {
			t.Fatalf("Load model error:\t error=%s, hash=%s\n", mErr, modelHash)
			return
		}

		// Predict
		start = time.Now()
		_, pErr := Predict(network, inputBufferList[i%length])
		predict_du += time.Since(start)
		if pErr != nil {
			t.Errorf("Predict error:\t error=%s, loop=%d, hash=%s\n", pErr, i, inputHashList[i%length])
		} else {
			// t.Logf("Predict succeed:\t label=%d, hash=%s\n", label, inputHashList[i%length])
		}

		// Free model
		start = time.Now()
		FreeModel(network)
		free_du += time.Since(start)
	}

	t.Logf("Load model\t loop=%d, elapsed=%vs\n", loop, load_du.Seconds())
	t.Logf("Predict model&input\t loop=%d, elapsed=%vs\n", loop, predict_du.Seconds())
	t.Logf("Free model\t loop=%d, elapsed=%vs\n", loop, free_du.Seconds())

	t.Fatal("Test ended")

}
