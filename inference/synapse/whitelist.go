package synapse

import (
	"errors"
	"fmt"
)

var (
	model_v1_whitelist = []string{
		"4d8bc8272b882f315c6a96449ad4568fac0e6038",
		"ca3d0286d5758697cdef653c1375960a868ac08a",
	}
)

var (
	Model_V1 = "model-v1"
)

func CheckMetaHash(version string, infoHash string) error {
	if version == Model_V1 {
		return checkModelVersionOne(infoHash)
	}

	return errors.New("Whitelist version Error")
}

func checkModelVersionOne(infoHash string) error {
	for _, hash := range model_v1_whitelist {
		if infoHash == hash {
			return nil
		}
	}

	return errors.New(fmt.Sprintf("Model %s error", infoHash))
}
