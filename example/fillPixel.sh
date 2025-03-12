ARGS=$1
CONTRACT=0x6859cb527b051de6c66405aa2fc89dd14ba97816
curl -X POST http://localhost:30089 -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"ctxc_sendTransaction","params":[{"from": "0xa96120378fb2c0bab24489902c7053351e023fc6", "to": "'$CONTRACT'", "input": "0x7468847d'$ARGS'", "price":"0x1", "gas":"0x101200"}], "id":"0x1"}'
