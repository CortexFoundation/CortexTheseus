package inference

// Inference Engine Interface
type Inference interface {
	VerifyModel(modelInfoHash string) error
	VerifyInput(inputInfoHash string) error

	// Infer procedure may be blocked with I/O, infernet, network, etc.
	// It should be designed with returning result to make thread be killed internal possible
	InferByInfoHash(modelInfoHash, inputInfoHash string) (uint64, error)
	InferByInputContent(modelInfoHash string, inputContent []byte) (uint64, error)

	RemoteInferByInfoHash(modelInfoHash, inputInfoHash, uri string) (uint64, error)
	RemoteInferByInputContent(modelInfoHash, uri string, addr, slot, blockNumber string) (uint64, error)
}
