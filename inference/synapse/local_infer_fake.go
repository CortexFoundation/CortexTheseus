// +build remote

package synapse

import "errors"

func (s *Synapse) InferByInfoHash(modelInfoHash, inputInfoHash string) ([]byte, error) {
	return nil, errors.New("LocalInfer not implemented")
}

func (s *Synapse) InferByInputContent(modelInfoHash string, inputContent []byte) ([]byte, error) {
	return nil, errors.New("LocalInfer not implemented")
}

func (s *Synapse) GetGasByInfoHash(modelInfoHash string) (gas uint64, err error) {
	return 0, errors.New("LocalInfer not implemented")
}
