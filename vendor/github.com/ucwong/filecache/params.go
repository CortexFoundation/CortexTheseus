package filecache

import (
	"runtime"
)

// File size constants for use with FileCache.MaxSize.
// For example, cache.MaxSize = 64 * Megabyte
const (
	Kilobyte = 1024
	Megabyte = 1024 * 1024
	Gigabyte = 1024 * 1024 * 1024
)

var (
	SquelchItemNotInCache bool = true

	DefaultExpireItem int = 300 // 5 minutes

	// Max size for each item
	DefaultMaxSize int64 = 16 * Megabyte

	// Max amount of items
	DefaultMaxItems int = 32

	// Check interval by seconds
	DefaultEvery int = 60 // 1 minute

	// Mumber of items to buffer adding to the file cache.
	NewCachePipeSize int = runtime.NumCPU() * 8
)
