package common

// Block ... block struct
type Block struct {
	Number     string
	Hash       string
	ParentHash string
	Txs        []map[string]string `json:"Transactions"`
}

// TransactionReceipt ...
type TransactionReceipt struct {
	ContractAddr string `json:"ContractAddr"`
	TxHash       string `json:"TransactionHash"`
}

// FileMeta ...
type FileMeta struct {
	// Transaction hash
	TxHash       string
	ContractAddr string
	AuthorAddr   string
	URI          string
	RawSize      uint64
	BlockNum     uint64
}

// FileStorage ...
type FileStorage struct {
}

// FlowControlMeta ...
type FlowControlMeta struct {
	URI            string
	BytesRequested uint64
}
