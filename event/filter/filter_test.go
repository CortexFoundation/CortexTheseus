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

package filter

import (
	"testing"
	"time"
)

// Simple test to check if baseline matching/mismatching filtering works.
func TestFilters(t *testing.T) {
	fm := New()
	fm.Start()

	// Register two filters to catch posted data
	first := make(chan struct{})
	fm.Install(Generic{
		Str1: "hello",
		Fn: func(data any) {
			first <- struct{}{}
		},
	})
	second := make(chan struct{})
	fm.Install(Generic{
		Str1: "hello1",
		Str2: "hello",
		Fn: func(data any) {
			second <- struct{}{}
		},
	})
	// Post an event that should only match the first filter
	fm.Notify(Generic{Str1: "hello"}, true)
	fm.Stop()

	// Ensure only the mathcing filters fire
	select {
	case <-first:
	case <-time.After(100 * time.Millisecond):
		t.Error("matching filter timed out")
	}
	select {
	case <-second:
		t.Error("mismatching filter fired")
	case <-time.After(100 * time.Millisecond):
	}
}
