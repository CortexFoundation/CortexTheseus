# Folder Storing Data and Scripts for Testing t8n Tools

## without signing(v r s already generated)
`./cvm t8n --input.alloc=./t8ntool/t8n_test/testdata/1/alloc.json --input.txs=./t8ntool/t8n_test/testdata/1/txs.json --input.env=./testdata/1/env.json --output.result=stdout --output.alloc=stdout`

## with signing(v r s 0x0, secretKey set)
`./cvm t8n --input.alloc=./t8ntool/t8n_test/testdata/1/alloc.json --input.txs=./t8ntool/t8n_test/testdata/1/txs_sec.json --input.env=./testdata/1/env.json --output.result=stdout --output.alloc=stdout`
