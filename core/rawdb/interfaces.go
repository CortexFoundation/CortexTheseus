// Copyright 2018 The CortexTheseus Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package rawdb

// DatabaseReader wraps the Has and Get method of a backing data store.
type DatabaseReader interface {
	Has(key []byte) (bool, error)
	Get(key []byte) ([]byte, error)
}

// DatabaseWriter wraps the Put method of a backing data store.
type DatabaseWriter interface {
	Put(key []byte, value []byte) error
}

// DatabaseDeleter wraps the Delete method of a backing data store.
type DatabaseDeleter interface {
	Delete(key []byte) error
}

/*type AncientWriter interface {
        // AppendAncient injects all binary blobs belong to block at the end of the
        // append-only immutable table files.
        AppendAncient(number uint64, hash, header, body, receipt, td []byte) error

        // TruncateAncients discards all but the first n ancient data from the ancient store.
        TruncateAncients(n uint64) error

        // Sync flushes all in-memory ancient store data to disk.
        Sync() error
}*/

/*type KeyValueStore interface {
        KeyValueReader
        KeyValueWriter
        Batcher
        Iteratee
        Stater
        Compacter
        io.Closer
}*/
