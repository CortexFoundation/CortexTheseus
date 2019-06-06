// Copyright 2017 The go-cortex Authors
// This file is part of the go-cortex library.
//
// The go-cortex library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-cortex library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the go-cortex library. If not, see <http://www.gnu.org/licenses/>.

package params

// These are network parameters that need to be constant between clients, but
// aren't necessarily consensus related.

const (
	// BloomBitsBlocks is the number of blocks a single bloom bit section vector
	// contains on the server side.
	BloomBitsBlocks uint64 = 4096

	// BloomConfirms is the number of confirmation blocks before a bloom section is
	// considered probably final and its rotated bits are calculated.
	BloomConfirms = 256

	CHTFrequency = 65536
	/*Check point:1 65535
	Check point:2 131071
	Check point:3 196607
	Check point:4 262143
	Check point:5 327679
	Check point:6 393215
	Check point:7 458751
	Check point:8 524287
	Check point:9 589823
	Check point:10 655359
	Check point:11 720895
	Check point:12 786431
	Check point:13 851967
	Check point:14 917503
	Check point:15 983039
	Check point:16 1048575
	Check point:17 1114111
	Check point:18 1179647
	Check point:19 1245183
	Check point:20 1310719*/
)
