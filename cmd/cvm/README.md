# An Indivisual Main Entry For CVM

## modified files:
cmd/cvm/*
core/state/dump_collector.go
core/vm/logger_json.go


## How to run Infer test:
1. prepare plugins/libcvm_runtime.so

2. prepare test model files
```
tf_data
|-- 0000000000000000000000000000000000001013
|   '-- data
|       |-- params
|       '-- symbol
'-- 0000000000000000000000000000000000002013
    '-- data
```

3. modify torrentfs instance(fs.go) to read files with dataDir, infoHash and subPath (when using torrentfs/localfile, no need to modify instance, just set several config lines)
``` go
func (fs *TorrentFS) GetFileWithSize(ctx context.Context, infohash string, rawSize uint64, subpath string) ([]byte, error) {
    return fs.GetFile(ctx, infohash, subpath)
}
// GetFile is used to get file from storage, current this will not be call after available passed
func (fs *TorrentFS) GetFile(ctx context.Context, infohash, subpath string) ([]byte, error) {
    return ioutil.ReadFile(fs.storage().DataDir + "/" + infohash + subpath)
}
```

4. insert log stdout for viewing opcode, stack and memory in Run(core/vm/interpreter.go) and opInfer(core/vm/instructions.go)

5. compile solidity infer with ctxc-solc-v2 and get hexcode:
`ctxc-solc-v2 --bin $(SOLFILENAME)`

6. run test under cmd/cvm with:
`go build && ./cvm --code <hexcode> run`
or run with default tracer with detailed debug msg:
`go build && ./cvm --code <hexcode> --debug run`


## Result:
1. infer: model(cvm_mnist) input(data) output[129 211 178 164 214 183 137 72 129 200](Complement)
2. inferArray: model(cvm_mnist) input(uint8[1,1,28,28]{0}) output[255 1 255 255 0 255 255 0 0 0](Complement)

## Transition Tool::
1. alloc.json allocates accounts(better with secretKey), account gen by Generate
2. txs.json puts transactions, may not be signed with secretKey of Sender Account
3. run demo script: "./testdata/test.sh"

