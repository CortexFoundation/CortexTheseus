// Copyright 2022 The CortexTheseus Authors
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
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>

package wormhole

var Wormholes = []string{
	"http://wormhole.cortexlabs.ai:30089/tunnel?hash=",
}

const (
	CAP = 20
)

var (
	BestTrackerUrl = []string{
		"https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best.txt",
		"https://cdn.jsdelivr.net/gh/ngosang/trackerslist@master/trackers_best.txt",
		"https://ngosang.github.io/trackerslist/trackers_best.txt",

		"https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_all.txt",
		"https://ngosang.github.io/trackerslist/trackers_all.txt",
		"https://cdn.jsdelivr.net/gh/ngosang/trackerslist@master/trackers_all.txt",

		// ips
		"https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best_ip.txt",
		"https://ngosang.github.io/trackerslist/trackers_best_ip.txt",
		"https://cdn.jsdelivr.net/gh/ngosang/trackerslist@master/trackers_best_ip.txt",

		"https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_all_ip.txt",
		"https://ngosang.github.io/trackerslist/trackers_all_ip.txt",
		"https://cdn.jsdelivr.net/gh/ngosang/trackerslist@master/trackers_all_ip.txt",
	}

	AllTrackerUrl = []string{
		"https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_all.txt",
		"https://ngosang.github.io/trackerslist/trackers_all.txt",
		"https://cdn.jsdelivr.net/gh/ngosang/trackerslist@master/trackers_all.txt",
	}

	IPTrackerUrl = []string{
		"https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_best_ip.txt",
		"https://ngosang.github.io/trackerslist/trackers_best_ip.txt",
		"https://cdn.jsdelivr.net/gh/ngosang/trackerslist@master/trackers_best_ip.txt",

		"https://raw.githubusercontent.com/ngosang/trackerslist/master/trackers_all_ip.txt",
		"https://ngosang.github.io/trackerslist/trackers_all_ip.txt",
		"https://cdn.jsdelivr.net/gh/ngosang/trackerslist@master/trackers_all_ip.txt",
	}

	ColaUrl = []string{"https://github.com/CortexFoundation/cola/releases/download/1.0.0/cola.txt"}
)
