//go:build !linksqlite3
// +build !linksqlite3

// Package c contains only a C file.
//
// This Go file is part of a workaround for `go mod vendor`.
// Please see the file dummy.go at the root of the module for more information.
package c

/*
// !!! UPDATE THE Makefile WITH THESE DEFINES !!!
#cgo CFLAGS: -DSQLITE_THREADSAFE=2
#cgo CFLAGS: -DSQLITE_DEFAULT_WAL_SYNCHRONOUS=1
#cgo CFLAGS: -DSQLITE_ENABLE_UNLOCK_NOTIFY
#cgo CFLAGS: -DSQLITE_ENABLE_FTS5
#cgo CFLAGS: -DSQLITE_ENABLE_RTREE
#cgo CFLAGS: -DSQLITE_LIKE_DOESNT_MATCH_BLOBS
#cgo CFLAGS: -DSQLITE_OMIT_DEPRECATED
#cgo CFLAGS: -DSQLITE_ENABLE_JSON1
#cgo CFLAGS: -DSQLITE_ENABLE_SESSION
#cgo CFLAGS: -DSQLITE_ENABLE_SNAPSHOT
#cgo CFLAGS: -DSQLITE_ENABLE_PREUPDATE_HOOK
#cgo CFLAGS: -DSQLITE_USE_ALLOCA
#cgo CFLAGS: -DSQLITE_ENABLE_COLUMN_METADATA
#cgo CFLAGS: -DHAVE_USLEEP=1
#cgo CFLAGS: -DSQLITE_DQS=0
#cgo CFLAGS: -DSQLITE_ENABLE_GEOPOLY
#cgo CFLAGS: -DSQLITE_DIRECT_OVERFLOW_READ
#cgo windows LDFLAGS: -lwinpthread
#cgo linux LDFLAGS: -ldl -lm
#cgo linux CFLAGS: -std=c99
#cgo openbsd LDFLAGS: -lm
#cgo openbsd CFLAGS: -std=c99
#cgo freebsd LDFLAGS: -lm
#cgo freebsd CFLAGS: -std=c99
// !!! UPDATE THE Makefile WITH THESE DEFINES !!!
*/
import "C"
