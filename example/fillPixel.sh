ARGS=$1
CONTRACT=0xf3d04c84ee4a2d1a9417ed71a8f511d546963e0c
curl -X POST http://localhost:30089 -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"eth_sendTransaction","params":[{"from": "0xa96120378fb2c0bab24489902c7053351e023fc6", "to": "'$CONTRACT'", "input": "0x7468847d'$ARGS'", "price":"0x1", "gas":"0x1d1200"}], "id":"0x1"}'
