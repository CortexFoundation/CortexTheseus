package kv

import (
	"github.com/ucwong/golang-kv/badger"
	"github.com/ucwong/golang-kv/bolt"
	"github.com/ucwong/golang-kv/ha"
	"github.com/ucwong/golang-kv/leveldb"
)

func Badger(path string) Bucket {
	return badger.Open(path)
}

func Bolt(path string) Bucket {
	return bolt.Open(path)
}

func LevelDB(path string) Bucket {
	return leveldb.Open(path)
}

func HA(path string, level int) Bucket {
	return ha.Open(path, level)
}
