package kv

import (
	"time"
)

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
}
