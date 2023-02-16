# Golang-kv
Bundle embedded databases with fixed api https://pkg.go.dev/github.com/ucwong/golang-kv

Interfaces

```
type Bucket interface {
	Get(k []byte) []byte
	Set(k, v []byte) error
	Del(k []byte) error
	Prefix(k []byte) [][]byte
	Suffix(k []byte) [][]byte
	Scan() [][]byte
	Range(start, limit []byte) [][]byte
	SetTTL(k, v []byte, expire time.Duration) error
	Close() error

	// BatchSet write & flush
	BatchSet(kvs map[string][]byte) error
}
```

used by 
```
import "github.com/ucwong/golang-kv"

...

badger := kv.Badger("")
defer badger.Close()
badger.Set([]byte("x"), []byte("y")))
v := badger.Get([]byte("x"))
vs := badger.Prefix([]byte("x"))

...

bolt := kv.Bolt("")
defer bolt.Close()

bolt.setTTL([]byte("k"), []byte("v"), time.Second)

...

ldb := kv.LevelDB("")
defer ldb.Close()

...

peb := kv.Pebble("")
defer peb.Close()

...
```
## Test
```
make test
```
## Repobeats

![Alt](https://repobeats.axiom.co/api/embed/6187bb40cb7709dfbf08cd33cb06f44f046c7cb4.svg "Repobeats analytics image")

# How to choose database engine
![image](https://user-images.githubusercontent.com/22344498/111969569-5aede600-8b35-11eb-8580-8cd1baf2bbb1.png)

![image](https://user-images.githubusercontent.com/22344498/111968369-07c76380-8b34-11eb-90f3-26b0a2a85624.png)
