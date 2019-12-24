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
	/*Check section:18 622591
	Check section:19 655359
	Check section:20 688127
	Check section:21 720895
	Check section:22 753663
	Check section:23 786431
	Check section:24 819199
	Check section:25 851967
	Check section:26 884735
	Check section:27 917503
	Check section:28 950271
	Check section:29 983039
	Check section:30 1015807
	Check section:31 1048575
	Check section:32 1081343
	Check section:33 1114111
	Check section:34 1146879
	Check section:35 1179647
	Check section:36 1212415
	Check section:37 1245183
	Check section:38 1277951
	Check section:39 1310719
	Check section:40 1343487*/
	ImmutabilityThreshold = 90000
)
