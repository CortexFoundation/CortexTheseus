// Copyright 2021 Ross Light
// SPDX-License-Identifier: ISC

//go:build go1.16
// +build go1.16

package fs

import "io/fs"

// FS is an alias for the io/fs.FS interface.
type FS = fs.FS

// File is an alias for the io/fs.File interface.
type File = fs.File
