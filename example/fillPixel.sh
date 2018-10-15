ARGS=$1
CONTRACT=0x25befe823553cd37d93d91a44cf5f5c0ba568e99
curl -X POST http://localhost:30089 -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"eth_sendTransaction","params":[{"from": "0x432fee7c11Afc2F67C37b079aC284fc42Adee3bB", "to": "'$CONTRACT'", "input": "0x7468847d'$ARGS'", "price":"0x1", "gas":"0x3a1200"}], "id":"0x1"}'
