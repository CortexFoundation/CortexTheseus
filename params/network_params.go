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
			Check section:212 6979583
		Check section:213 7012351
		Check section:214 7045119
		Check section:215 7077887
		Check section:216 7110655
		Check section:217 7143423
		Check section:218 7176191
		Check section:219 7208959
		Check section:220 7241727
		Check section:221 7274495
		Check section:222 7307263
		Check section:223 7340031
		Check section:224 7372799
		Check section:225 7405567
		Check section:226 7438335
		Check section:227 7471103
		Check section:228 7503871
		Check section:229 7536639
		Check section:230 7569407
		Check section:231 7602175
		Check section:232 7634943
		Check section:233 7667711
		Check section:234 7700479
		Check section:235 7733247
		Check section:236 7766015
		Check section:237 7798783
		Check section:238 7831551
		Check section:239 7864319
		Check section:240 7897087
		Check section:241 7929855
		Check section:242 7962623
		Check section:243 7995391
		Check section:244 8028159
		Check section:245 8060927
		Check section:246 8093695
		Check section:247 8126463
		Check section:248 8159231
		Check section:249 8191999
		Check section:250 8224767
		Check section:251 8257535
		Check section:252 8290303
		Check section:253 8323071
		Check section:254 8355839
		Check section:255 8388607
		Check section:256 8421375
		Check section:257 8454143
		Check section:258 8486911
		Check section:259 8519679
		Check section:260 8552447
		Check section:261 8585215
		Check section:262 8617983
		Check section:263 8650751
		Check section:264 8683519
		Check section:265 8716287
		Check section:266 8749055
		Check section:267 8781823
		Check section:268 8814591
		Check section:269 8847359
		Check section:270 8880127
		Check section:271 8912895
		Check section:272 8945663
		Check section:273 8978431
		Check section:274 9011199
		Check section:275 9043967
		Check section:276 9076735
		Check section:277 9109503
		Check section:278 9142271
		Check section:279 9175039
		Check section:280 9207807
		Check section:281 9240575
		Check section:282 9273343
		Check section:283 9306111
		Check section:284 9338879
		Check section:285 9371647
		Check section:286 9404415
		Check section:287 9437183
		Check section:288 9469951
		Check section:289 9502719
		Check section:290 9535487
		Check section:291 9568255
		Check section:292 9601023
		Check section:293 9633791
		Check section:294 9666559
		Check section:295 9699327
		Check section:296 9732095
		Check section:297 9764863
	*/
	ImmutabilityThreshold     = 90000
	FullImmutabilityThreshold = 90000
)
