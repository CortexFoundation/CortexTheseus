// Copyright (C) 2022 ucwong
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>

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

	// BatchSet write & flush
	BatchSet(kvs map[string][]byte) error

	Name() string
}
