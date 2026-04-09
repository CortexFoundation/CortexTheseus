// Copyright 2026 The Cockroach Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

package markers

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"hash"
	"sync"
	"sync/atomic"
)

/*
	Hash function implementation notes:

	We use SHA-256 because it provides strong cryptographic properties with
	negligible performance overhead on modern hardwares.

	When a salt is provided via EnableHashing(), we use HMAC-SHA256 which
	provides additional security properties and domain separation.

	The hash output is truncated to 8 hex characters (32 bits) to keep log
	output concise while still providing sufficient collision resistance for
	typical logging workloads.

	Hasher instances are pooled via sync.Pool to avoid allocating a new
	SHA-256 or HMAC struct on every call. The pool is created once in
	EnableHashing() with the correct hasher type (SHA-256 or HMAC-SHA256)
	baked in, so hash functions just Get/Reset/Put without checking salt.
*/

// defaultHashLength is the number of hex characters to use from the hash.
// 8 hex chars = 32 bits = ~4.3 billion unique values.
// This provides a good balance between collision resistance and output brevity.
const defaultHashLength = 8

// compile-time assertion to guarantee defaultHashLength doesn't exceed hex-encoded hash length.
var _ [sha256.Size*2 - defaultHashLength]byte

// hasherState holds a reusable hash.Hash and a pre-allocated Sum buffer.
// sumBuf must live in the pool because Sum() is an interface method —
// the compiler can't prove it won't store a reference, because of hash being
// in pool, so a local [32]byte would escape to heap on every call.
// hexBuf is a stack-allocated local in the hash functions because
// hex.Encode is a concrete function call that doesn't cause escape.
type hasherState struct {
	h      hash.Hash
	sumBuf [sha256.Size]byte // reusable buffer for h.Sum() output
}

var hashConfig struct {
	enabled atomic.Bool
	pool    atomic.Value
}

// EnableHashing enables hash-based redaction with an optional salt.
// When salt is nil, hash markers use plain SHA-256.
// When salt is provided, hash markers use HMAC-SHA256 for better security.
//
// A sync.Pool of hasherState instances is created here with the correct
// hasher type, so hash functions just Get/Reset/Put without checking salt.
func EnableHashing(salt []byte) {
	var pool *sync.Pool
	if len(salt) > 0 {
		// Copy so the pool closure doesn't capture a mutable slice.
		saltCopy := make([]byte, len(salt))
		copy(saltCopy, salt)
		pool = &sync.Pool{
			New: func() interface{} {
				return &hasherState{h: hmac.New(sha256.New, saltCopy)}
			},
		}
	} else {
		pool = &sync.Pool{
			New: func() interface{} {
				return &hasherState{h: sha256.New()}
			},
		}
	}

	hashConfig.pool.Store(pool)
	hashConfig.enabled.Store(true)
}

// DisableHashing disables hash-based redaction.
// The pool and salt are intentionally left intact so that in-flight
// hashers (which captured the pool pointer) can finish safely.
func DisableHashing() {
	hashConfig.enabled.Store(false)
}

// IsHashingEnabled returns true if hash-based redaction is enabled.
func IsHashingEnabled() bool {
	return hashConfig.enabled.Load()
}

// appendHash computes a truncated hash of value and appends the hex result
// directly to dst, avoiding an intermediate allocation.
// Must only be called when hashing is enabled (IsHashingEnabled() == true).
func appendHash(dst []byte, value []byte) []byte {
	p := hashConfig.pool.Load().(*sync.Pool)
	state := p.Get().(*hasherState)

	var hexBuf [sha256.Size * 2]byte
	state.h.Reset()
	state.h.Write(value)
	sum := state.h.Sum(state.sumBuf[:0])
	hex.Encode(hexBuf[:], sum)

	dst = append(dst, hexBuf[:defaultHashLength]...)
	p.Put(state)
	return dst
}
