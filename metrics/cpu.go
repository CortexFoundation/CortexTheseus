// Copyright 2018 The CortexTheseus Authors
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

package metrics

import "github.com/elastic/gosigar"

// CPUStats is the system and process CPU stats.
type CPUStats struct {
	GlobalTime int64 // Time spent by the CPU working on all processes
	GlobalWait int64 // Time spent by waiting on disk for all processes
	LocalTime  int64 // Time spent by the CPU working on this process
}

// ReadCPUStats retrieves the current CPU stats.
func ReadCPUStats(stats *CPUStats) {
	global := gosigar.Cpu{}
	global.Get()

	stats.GlobalTime = int64(global.User + global.Nice + global.Sys)
	stats.GlobalWait = int64(global.Wait)
	stats.LocalTime = getProcessCPUTime()
}
