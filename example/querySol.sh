CONTRACT=$1 
SLOT=$2
BN=$3
curl -X POST -H "Content-Type:application/json" -d '{"jsonrpc":"2.0", "method": "ctxc_getSolidityBytes", "params": ["'$CONTRACT'", "'$SLOT'", "'$BN'"], "id": 1}' http://localhost:30089
