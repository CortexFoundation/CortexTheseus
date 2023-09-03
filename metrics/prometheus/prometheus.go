// Copyright 2019 The CortexTheseus Authors
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

// Package prometheus exposes go-metrics into a Prometheus format.
package prometheus

import (
	"fmt"
	"net/http"
	"sort"

	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/metrics"
)

// Handler returns an HTTP handler which dump metrics in Prometheus format.
func Handler(reg metrics.Registry) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Gather and pre-sort the metrics to avoid random listings
		var names []string
		reg.Each(func(name string, i interface{}) {
			names = append(names, name)
		})
		sort.Strings(names)

		// Aggregate all the metrics into a Prometheus collector
		c := newCollector()

		for _, name := range names {
			i := reg.Get(name)
			if err := c.Add(name, i); err != nil {
				log.Warn("Unknown Prometheus metric type", "type", fmt.Sprintf("%T", i))
			}
		}
		w.Header().Add("Content-Type", "text/plain")
		w.Header().Add("Content-Length", fmt.Sprint(c.buff.Len()))
		w.Write(c.buff.Bytes())
	})
}
