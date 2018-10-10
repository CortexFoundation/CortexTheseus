// +build remote

package synapse

import "errors"

func (s Synapse) InferByInfoHash(modelInfoHash, inputInfoHash string) (uint64, error) {
	return 0, errors.New("LocalInfer not implemented")
}

func (s Synapse) InferByInputContent(modelInfoHash string, inputContent []byte) (uint64, error) {
	return 0, errors.New("LocalInfer not implemented")
}
