# shard

## Test
```
make
```

## Benchmark
```
cd bench
```
```
make
```
```
go version go1.24.6 linux/amd64

     number of cpus: 2
     number of keys: 1000000
            keysize: 10
        random seed: 1755204165544372244

-- sync.Map --
set: 1,000,000 ops over 2 threads in 1636ms, 611,340/sec, 1635 ns/op
get: 1,000,000 ops over 2 threads in 596ms, 1,677,930/sec, 595 ns/op
rng:       100 ops over 2 threads in 14152ms, 7/sec, 141517584 ns/op
del: 1,000,000 ops over 2 threads in 1052ms, 950,852/sec, 1051 ns/op

-- stdlib map --
set: 1,000,000 ops over 2 threads in 1429ms, 699,759/sec, 1429 ns/op
get: 1,000,000 ops over 2 threads in 301ms, 3,318,098/sec, 301 ns/op
rng:       100 ops over 2 threads in 2201ms, 45/sec, 22012248 ns/op
del: 1,000,000 ops over 2 threads in 554ms, 1,804,653/sec, 554 ns/op

-- github.com/orcaman/concurrent-map/v2 --
set: 1,000,000 ops over 2 threads in 1542ms, 648,352/sec, 1542 ns/op
get: 1,000,000 ops over 2 threads in 468ms, 2,135,986/sec, 468 ns/op
rng:       100 ops over 2 threads in 30589ms, 3/sec, 305885363 ns/op
del: 1,000,000 ops over 2 threads in 633ms, 1,580,664/sec, 632 ns/op

-- shardmap --
set: 1,000,000 ops over 2 threads in 538ms, 1,859,975/sec, 537 ns/op
get: 1,000,000 ops over 2 threads in 233ms, 4,293,377/sec, 232 ns/op
rng:       100 ops over 2 threads in 1958ms, 51/sec, 19581166 ns/op
del: 1,000,000 ops over 2 threads in 449ms, 2,227,468/sec, 448 ns/op

goos: linux
goarch: amd64
pkg: github.com/ucwong/shard/bench
cpu: Intel(R) Xeon(R) Platinum 8163 CPU @ 2.50GHz
BenchmarkFib10     	       1	58952746944 ns/op
BenchmarkFib10-2   	
go version go1.24.6 linux/amd64

     number of cpus: 2
     number of keys: 1000000
            keysize: 10
        random seed: 1755204224503493950

-- sync.Map --
set: 1,000,000 ops over 2 threads in 1974ms, 506,531/sec, 1974 ns/op
get: 1,000,000 ops over 2 threads in 609ms, 1,642,803/sec, 608 ns/op
rng:       100 ops over 2 threads in 13690ms, 7/sec, 136896219 ns/op
del: 1,000,000 ops over 2 threads in 1168ms, 856,300/sec, 1167 ns/op

-- stdlib map --
set: 1,000,000 ops over 2 threads in 1609ms, 621,603/sec, 1608 ns/op
get: 1,000,000 ops over 2 threads in 368ms, 2,719,525/sec, 367 ns/op
rng:       100 ops over 2 threads in 2597ms, 38/sec, 25968226 ns/op
del: 1,000,000 ops over 2 threads in 816ms, 1,225,600/sec, 815 ns/op

-- github.com/orcaman/concurrent-map/v2 --
set: 1,000,000 ops over 2 threads in 1752ms, 570,660/sec, 1752 ns/op
get: 1,000,000 ops over 2 threads in 635ms, 1,575,485/sec, 634 ns/op
rng:       100 ops over 2 threads in 38060ms, 2/sec, 380598706 ns/op
del: 1,000,000 ops over 2 threads in 575ms, 1,738,674/sec, 575 ns/op

-- shardmap --
set: 1,000,000 ops over 2 threads in 650ms, 1,538,633/sec, 649 ns/op
get: 1,000,000 ops over 2 threads in 243ms, 4,117,897/sec, 242 ns/op
rng:       100 ops over 2 threads in 2068ms, 48/sec, 20677503 ns/op
del: 1,000,000 ops over 2 threads in 407ms, 2,455,728/sec, 407 ns/op

       1	67871461131 ns/op
BenchmarkFib10-4   	
go version go1.24.6 linux/amd64

     number of cpus: 2
     number of keys: 1000000
            keysize: 10
        random seed: 1755204292377974726

-- sync.Map --
set: 1,000,000 ops over 2 threads in 1389ms, 719,698/sec, 1389 ns/op
get: 1,000,000 ops over 2 threads in 545ms, 1,833,789/sec, 545 ns/op
rng:       100 ops over 2 threads in 11659ms, 8/sec, 116593918 ns/op
del: 1,000,000 ops over 2 threads in 1079ms, 926,737/sec, 1079 ns/op

-- stdlib map --
set: 1,000,000 ops over 2 threads in 1953ms, 512,158/sec, 1952 ns/op
get: 1,000,000 ops over 2 threads in 372ms, 2,687,461/sec, 372 ns/op
rng:       100 ops over 2 threads in 2653ms, 37/sec, 26531642 ns/op
del: 1,000,000 ops over 2 threads in 791ms, 1,263,577/sec, 791 ns/op

-- github.com/orcaman/concurrent-map/v2 --
set: 1,000,000 ops over 2 threads in 1464ms, 682,872/sec, 1464 ns/op
get: 1,000,000 ops over 2 threads in 509ms, 1,965,734/sec, 508 ns/op
rng:       100 ops over 2 threads in 33698ms, 2/sec, 336980533 ns/op
del: 1,000,000 ops over 2 threads in 549ms, 1,819,906/sec, 549 ns/op

-- shardmap --
set: 1,000,000 ops over 2 threads in 702ms, 1,424,359/sec, 702 ns/op
get: 1,000,000 ops over 2 threads in 268ms, 3,729,997/sec, 268 ns/op
rng:       100 ops over 2 threads in 1800ms, 55/sec, 18000083 ns/op
del: 1,000,000 ops over 2 threads in 420ms, 2,379,933/sec, 420 ns/op

       1	60409236203 ns/op
```
