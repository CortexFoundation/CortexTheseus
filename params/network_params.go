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

	CHTFrequency = 32768
	/*Check point:1 32767
	Check point:2 65535
	Check point:3 98303
	Check point:4 131071
	Check point:5 163839
	Check point:6 196607
	Check point:7 229375
	Check point:8 262143
	Check point:9 294911
	Check point:10 327679
	Check point:11 360447
	Check point:12 393215
	Check point:13 425983
	Check point:14 458751
	Check point:15 491519
	Check point:16 524287
	Check point:17 557055
	Check point:18 589823
	Check point:19 622591
	Check point:20 655359*/
)
