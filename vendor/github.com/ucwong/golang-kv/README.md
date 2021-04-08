# Golang-kv
Bundle embedded databases with fixed api

```
go run cmd/main.go
```

used by 
```
import "github.com/ucwong/golang-kv"

...

badger := kv.Badger("")
defer badger.Close()
badger.Set([]byte("x", []byte("y")))
v := badger.Get([]byte("x"))
vs := badger.Prefix([]byte("x"))

...

bolt := kv.Bolt("")
defer bolt.Close()

bolt.setTTL([]byte("k"), []byte("v"), time.Second)

...

ldb := kv.LevelDB("")
defer ldb.Close()

ha := kv.HA("")
defer ha.Close()

...

```
# How to choose database engine
![image](https://user-images.githubusercontent.com/22344498/111969569-5aede600-8b35-11eb-8580-8cd1baf2bbb1.png)

![image](https://user-images.githubusercontent.com/22344498/111968369-07c76380-8b34-11eb-90f3-26b0a2a85624.png)
