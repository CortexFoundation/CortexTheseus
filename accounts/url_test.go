// Copyright 2018 The go-ethereum Authors
// This file is part of The go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package accounts

import (
	"testing"
)

func TestURLParsing(t *testing.T) {
	t.Parallel()
	url, err := parseURL("https://CortexFoundation.org")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if url.Scheme != "https" {
		t.Errorf("expected: %v, got: %v", "https", url.Scheme)
	}
	if url.Path != "CortexFoundation.org" {
		t.Errorf("expected: %v, got: %v", "CortexFoundation.org", url.Path)
	}

	_, err = parseURL("CortexFoundation.org")
	if err == nil {
		t.Error("expected err, got: nil")
	}
}

func TestURLString(t *testing.T) {
	t.Parallel()
	url := URL{Scheme: "https", Path: "CortexFoundation.org"}
	if url.String() != "https://CortexFoundation.org" {
		t.Errorf("expected: %v, got: %v", "https://CortexFoundation.org", url.String())
	}

	url = URL{Scheme: "", Path: "CortexFoundation.org"}
	if url.String() != "CortexFoundation.org" {
		t.Errorf("expected: %v, got: %v", "CortexFoundation.org", url.String())
	}
}

func TestURLMarshalJSON(t *testing.T) {
	t.Parallel()
	url := URL{Scheme: "https", Path: "CortexFoundation.org"}
	json, err := url.MarshalJSON()
	if err != nil {
		t.Errorf("unexpcted error: %v", err)
	}
	if string(json) != "\"https://CortexFoundation.org\"" {
		t.Errorf("expected: %v, got: %v", "\"https://CortexFoundation.org\"", string(json))
	}
}

func TestURLUnmarshalJSON(t *testing.T) {
	t.Parallel()
	url := &URL{}
	err := url.UnmarshalJSON([]byte("\"https://CortexFoundation.org\""))
	if err != nil {
		t.Errorf("unexpcted error: %v", err)
	}
	if url.Scheme != "https" {
		t.Errorf("expected: %v, got: %v", "https", url.Scheme)
	}
	if url.Path != "CortexFoundation.org" {
		t.Errorf("expected: %v, got: %v", "https", url.Path)
	}
}

func TestURLComparison(t *testing.T) {
	t.Parallel()
	tests := []struct {
		urlA   URL
		urlB   URL
		expect int
	}{
		{URL{"https", "CortexFoundation.org"}, URL{"https", "CortexFoundation.org"}, 0},
		{URL{"http", "CortexFoundation.org"}, URL{"https", "CortexFoundation.org"}, -1},
		{URL{"https", "CortexFoundation.org/a"}, URL{"https", "CortexFoundation.org"}, 1},
		{URL{"https", "abc.org"}, URL{"https", "CortexFoundation.org"}, 1},
	}

	for i, tt := range tests {
		result := tt.urlA.Cmp(tt.urlB)
		if result != tt.expect {
			t.Errorf("test %d: cmp mismatch: expected: %d, got: %d", i, tt.expect, result)
		}
	}
}
