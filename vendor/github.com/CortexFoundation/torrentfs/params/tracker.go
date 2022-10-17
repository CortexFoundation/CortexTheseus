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
package params

var (
	MainnetTrackers = []string{
		"udp://tracker.cortexlabs.ai:5008",
		//"http://tracker.cortexlabs.ai:5008/announce",
		//"http://tracker.openbittorrent.com:80/announce",
		//"udp://tracker.openbittorrent.com:6969/announce",
		//"udp://tracker.opentrackr.org:1337/announce",
	}

	GlobalTrackers = []string{
		"udp://tracker.openbittorrent.com:6969/announce",
		"udp://tracker.opentrackr.org:1337/announce",
	}
	/*MainnetTrackers = [][]string{
	        {"udp://tracker.cortexlabs.ai:5008"},
	        {"udp://tracker.openbittorrent.com:80"},
	        {"udp://tracker.publicbt.com:80"},
	        {"udp://tracker.istole.it:6969"},
	}*/

	//BernardTrackers = MainnetTrackers

	TorrentBoostNodes = []string{
		"http://share.cortexlabs.ai:7881",
	}

	BestTrackerUrl = []string{"https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best.txt",
		"https://cdn.jsdelivr.net/gh/ngosang/trackerslist@master/trackers_best.txt",
		"https://ngosang.github.io/trackerslist/trackers_best.txt",
	}
)
