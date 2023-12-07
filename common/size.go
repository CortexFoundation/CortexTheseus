// Copyright 2018 The go-ethereum Authors
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

package common

import (
	"fmt"
)

// StorageSize is a wrapper around a float value that supports user friendly
// formatting.
type StorageSize float64

// String implements the stringer interface.
func (s StorageSize) String() string {
	if s > 1099511627776 {
		return fmt.Sprintf("%.2f TiB", s/1099511627776)
	} else if s > 1073741824 {
		return fmt.Sprintf("%.2f GiB", s/1073741824)
	} else if s > 1048576 {
		return fmt.Sprintf("%.2f MiB", s/1048576)
	} else if s > 1024 {
		return fmt.Sprintf("%.2f KiB", s/1024)
	} else {
		return fmt.Sprintf("%.2f B", s)
	}
}

// TerminalString implements log.TerminalStringer, formatting a string for console
// output during logging.
func (s StorageSize) TerminalString() string {
	if s > 1099511627776 {
		return fmt.Sprintf("%.2fTiB", s/1099511627776)
	} else if s > 1073741824 {
		return fmt.Sprintf("%.2fGiB", s/1073741824)
	} else if s > 1048576 {
		return fmt.Sprintf("%.2fMiB", s/1048576)
	} else if s > 1024 {
		return fmt.Sprintf("%.2fKiB", s/1024)
	} else {
		return fmt.Sprintf("%.2fB", s)
	}
}

type HashSize float64

// String implements the stringer interface.
func (s HashSize) String() string {
	if s > 1099511627776 {
		return fmt.Sprintf("%.3f TiH", s/1099511627776)
	} else if s > 1073741824 {
		return fmt.Sprintf("%.3f GiH", s/1073741824)
	} else if s > 1048576 {
		return fmt.Sprintf("%.3f MiH", s/1048576)
	} else if s > 1024 {
		return fmt.Sprintf("%.3f KiH", s/1024)
	} else {
		return fmt.Sprintf("%.3f H", s)
	}
}

// TerminalString implements log.TerminalStringer, formatting a string for console
// output during logging.
func (s HashSize) TerminalString() string {
	if s > 1099511627776 {
		return fmt.Sprintf("%.3fTiH", s/1099511627776)
	} else if s > 1073741824 {
		return fmt.Sprintf("%.3fGiH", s/1073741824)
	} else if s > 1048576 {
		return fmt.Sprintf("%.3fMiH", s/1048576)
	} else if s > 1024 {
		return fmt.Sprintf("%.3fKiH", s/1024)
	} else {
		return fmt.Sprintf("%.3fH", s)
	}
}
