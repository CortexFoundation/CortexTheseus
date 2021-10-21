// Copyright 2020 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

package torrentfs

import (
	"context"
)

type CortexStorage interface {
	//Available(ctx context.Context, infohash string, rawSize uint64) (bool, error)
	//GetFile(ctx context.Context, infohash, path string) ([]byte, error)
	GetFileWithSize(ctx context.Context, infohash string, rawSize uint64, path string) ([]byte, error)
	Stop() error

	Download(ctx context.Context, ih string, request uint64) error
	SeedingLocal(ctx context.Context, filePath string, isLinkMode bool) (string, error)

	//0 finish, 1 pending, 2 downloading, 3 none
	Status(ctx context.Context, ih string) (int, error)
}
