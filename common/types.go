package common

//go:generate gencodec -type Header -field-override headerMarshaling -out gen_header_json.go

// Block ... block struct
type Block struct {
	Number     string
	Hash       string
	ParentHash string
	Txs        []map[string]string `json:"Transactions"`
}

// TransactionReceipt ...
type TransactionReceipt struct {
	ContractAddress string
	TransactionHash string
}

// FileMeta ...
type FileMeta struct {
	// Transaction hash
	TxHash string
	// transaction address
	TxAddress     string
	AuthorAddress string
	URI           string
	RawSize       uint64
	BlockNum      uint64
}

// FlowControlMeta ...
type FlowControlMeta struct {
	URI            string
	BytesRequested uint64
}
