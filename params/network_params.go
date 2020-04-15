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
	/*Check section:41 1376255
	Check section:42 1409023
	Check section:43 1441791
	Check section:44 1474559
	Check section:45 1507327
	Check section:46 1540095
	Check section:47 1572863
	Check section:48 1605631
	Check section:49 1638399
	Check section:50 1671167
	Check section:51 1703935
	Check section:52 1736703
	Check section:53 1769471
	Check section:54 1802239
	Check section:55 1835007
	Check section:56 1867775
	Check section:57 1900543
	Check section:58 1933311
	Check section:59 1966079
	Check section:60 1998847
	*/
	ImmutabilityThreshold = 90000
)
