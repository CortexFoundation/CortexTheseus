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
		Check section:100 3309567
		Check section:101 3342335
		Check section:102 3375103
		Check section:103 3407871
		Check section:104 3440639
		Check section:105 3473407
		Check section:106 3506175
		Check section:107 3538943
		Check section:108 3571711
		Check section:109 3604479
		Check section:110 3637247
		Check section:111 3670015
		Check section:112 3702783
		Check section:113 3735551
		Check section:114 3768319
		Check section:115 3801087
		Check section:116 3833855
		Check section:117 3866623
		Check section:118 3899391
		Check section:119 3932159
		Check section:120 3964927
		Check section:121 3997695
		Check section:122 4030463
		Check section:123 4063231
		Check section:124 4095999
		Check section:125 4128767
		Check section:126 4161535
		Check section:127 4194303
		Check section:128 4227071
		Check section:129 4259839
		Check section:130 4292607
		Check section:131 4325375
		Check section:132 4358143
		Check section:133 4390911
		Check section:134 4423679
		Check section:135 4456447
		Check section:136 4489215
		Check section:137 4521983
		Check section:138 4554751
		Check section:139 4587519
		Check section:140 4620287
		Check section:141 4653055
		Check section:142 4685823
		Check section:143 4718591
		Check section:144 4751359
		Check section:145 4784127
		Check section:146 4816895
		Check section:147 4849663
		Check section:148 4882431
		Check section:149 4915199
		Check section:150 4947967
		Check section:151 4980735
		Check section:152 5013503
		Check section:153 5046271
		Check section:154 5079039
		Check section:155 5111807
		Check section:156 5144575
		Check section:157 5177343
		Check section:158 5210111
		Check section:159 5242879
		Check section:160 5275647
		Check section:161 5308415
		Check section:162 5341183
		Check section:163 5373951
		Check section:164 5406719
		Check section:165 5439487
		Check section:166 5472255
		Check section:167 5505023
		Check section:168 5537791
		Check section:169 5570559
		Check section:170 5603327
		Check section:171 5636095
		Check section:172 5668863
		Check section:173 5701631
		Check section:174 5734399
		Check section:175 5767167
		Check section:176 5799935
		Check section:177 5832703
		Check section:178 5865471
		Check section:179 5898239
		Check section:180 5931007
		Check section:181 5963775
		Check section:182 5996543
		Check section:183 6029311
		Check section:184 6062079
		Check section:185 6094847
		Check section:186 6127615
		Check section:187 6160383
		Check section:188 6193151
		Check section:189 6225919
		Check section:190 6258687
		Check section:191 6291455
		Check section:192 6324223
		Check section:193 6356991
		Check section:194 6389759
		Check section:195 6422527
		Check section:196 6455295
		Check section:197 6488063

	*/
	ImmutabilityThreshold     = 90000
	FullImmutabilityThreshold = 90000
)
