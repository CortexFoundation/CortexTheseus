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
	/*
		Check section:60 1998847
		Check section:61 2031615
		Check section:62 2064383
		Check section:63 2097151
		Check section:64 2129919
		Check section:65 2162687
		Check section:66 2195455
		Check section:67 2228223
		Check section:68 2260991
		Check section:69 2293759
		Check section:70 2326527
		Check section:71 2359295
		Check section:72 2392063
		Check section:73 2424831
		Check section:74 2457599
		Check section:75 2490367
		Check section:76 2523135
		Check section:77 2555903
		Check section:78 2588671
		Check section:79 2621439
		Check section:80 2654207
		Check section:81 2686975
		Check section:82 2719743
		Check section:83 2752511
		Check section:84 2785279
		Check section:85 2818047
		Check section:86 2850815
		Check section:87 2883583
		Check section:88 2916351
		Check section:89 2949119
		Check section:90 2981887
		Check section:91 3014655
		Check section:92 3047423
		Check section:93 3080191
		Check section:94 3112959
		Check section:95 3145727
		Check section:96 3178495
		Check section:97 3211263
		Check section:98 3244031
	*/
	ImmutabilityThreshold = 90000
)
