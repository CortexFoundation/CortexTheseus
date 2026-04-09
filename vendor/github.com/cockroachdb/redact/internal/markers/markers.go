// Copyright 2020 The Cockroach Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

package markers

import (
	"bytes"
	"strings"

	i "github.com/cockroachdb/redact/interfaces"
)

// RedactableString is a string that contains a mix of safe and unsafe
// bits of data, but where it is known that unsafe bits are enclosed
// by redaction markers ‹ and ›, and occurrences of the markers
// inside the original data items have been escaped.
//
// Instances of RedactableString should not be constructed directly;
// instead use the facilities from print.go (Sprint, Sprintf)
// or the methods below.
type RedactableString string

// StripMarkers removes the redaction markers from the
// RedactableString. This returns an unsafe string where all safe and
// unsafe bits are mixed together.
func (s RedactableString) StripMarkers() string {
	// Avoid the []byte conversion when no markers are present.
	// All marker characters share the same leading UTF-8 byte (0xE2).
	// This may false-positive on other Unicode characters (€, —, etc.),
	// but stripMarkersBytes will correctly leave them untouched.
	if strings.IndexByte(string(s), StartBytes[0]) == -1 {
		return string(s)
	}
	return string(stripMarkersBytes([]byte(s), nil))
}

// Redact replaces all occurrences of unsafe substrings by the
// "Redacted" marker, ‹×›. Hash markers (‹†value›) are replaced
// with hashed values (‹hash›) if hashing is enabled, otherwise
// they are redacted like regular markers. The result string is still safe.
func (s RedactableString) Redact() RedactableString {
	if !strings.Contains(string(s), StartS) {
		return s
	}
	return RedactableString(redactBytes([]byte(s)))
}

// ToBytes converts the string to a byte slice.
func (s RedactableString) ToBytes() RedactableBytes {
	return RedactableBytes([]byte(string(s)))
}

// SafeFormat formats the redactable safely.
func (s RedactableString) SafeFormat(sp i.SafePrinter, _ rune) {
	// As per annotateArgs() in markers_internal_print.go,
	// we consider the redactable string not further formattable.
	sp.Print(s)
}

// RedactableBytes is like RedactableString but is a byte slice.
//
// Instances of RedactableBytes should not be constructed directly;
// instead use the facilities from print.go (Sprint, Sprintf)
// or the methods below.
type RedactableBytes []byte

// StripMarkers removes the redaction markers from the
// RedactableBytes. This returns an unsafe string where all safe and
// unsafe bits are mixed together.
func (s RedactableBytes) StripMarkers() []byte {
	return stripMarkersBytes([]byte(s), nil)
}

// Redact replaces all occurrences of unsafe substrings by the
// "Redacted" marker, ‹×›. Hash markers (‹†value›) are replaced
// with hashed values (‹hash›) if hashing is enabled, otherwise
// they are redacted like regular markers.
func (s RedactableBytes) Redact() RedactableBytes {
	return RedactableBytes(redactBytes([]byte(s)))
}

// redactBytes is the shared implementation for both RedactableString.Redact
// and RedactableBytes.Redact.
func redactBytes(data []byte) []byte {
	// Fast path: no markers at all.
	idx := bytes.Index(data, StartBytes)
	if idx == -1 {
		return data
	}
	hashEnabled := IsHashingEnabled()
	// len(data) is exact for the non-hash path (markers always shrink) and a
	// close lower bound for the hash path. Hash markers with content shorter
	// than 5 bytes expand slightly (e.g. ‹†x› 10B → ‹abcdef01› 14B), but
	// this is rare enough that letting append grow is cheaper than a pre-scan.
	buf := make([]byte, 0, len(data))
	pos := 0
	for idx != -1 {
		buf = append(buf, data[pos:pos+idx]...)
		markerStart := pos + idx
		contentStart := markerStart + StartLen
		j := bytes.Index(data[contentStart:], EndBytes)
		if j == -1 {
			buf = append(buf, data[markerStart:]...)
			return buf
		}
		if hashEnabled && j >= len(HashPrefixBytes) &&
			bytes.Equal(data[contentStart:contentStart+len(HashPrefixBytes)], HashPrefixBytes) {
			value := data[contentStart+len(HashPrefixBytes) : contentStart+j]
			buf = append(buf, StartBytes...)
			buf = appendHash(buf, value)
			buf = append(buf, EndBytes...)
		} else {
			buf = append(buf, RedactedBytes...)
		}
		pos = contentStart + j + EndLen
		idx = bytes.Index(data[pos:], StartBytes)
	}
	buf = append(buf, data[pos:]...)
	return buf
}

// ToString converts the byte slice to a string.
func (s RedactableBytes) ToString() RedactableString {
	return RedactableString(string([]byte(s)))
}

// SafeFormat formats the redactable safely.
func (s RedactableBytes) SafeFormat(sp i.SafePrinter, _ rune) {
	// As per annotateArgs() in markers_internal_print.go,
	// we consider the redactable bytes not further formattable.
	sp.Print(s)
}

// StartMarker returns the start delimiter for an unsafe string.
func StartMarker() []byte { return []byte(StartS) }

// EndMarker returns the end delimiter for an unsafe string.
func EndMarker() []byte { return []byte(EndS) }

// RedactedMarker returns the special string used by Redact.
func RedactedMarker() []byte { return []byte(RedactedS) }

// EscapeMarkers escapes the special delimiters from the provided
// byte slice.
func EscapeMarkers(s []byte) []byte {
	return stripMarkersBytes(s, EscapeMarkBytes)
}

// markerLen is the UTF-8 byte length of the marker characters.
// All marker characters (‹, ›, †) are 3-byte UTF-8 sequences sharing
// the same first two bytes.
const markerLen = 3

func init() {
	// Verify that all marker characters share the same 2-byte UTF-8 prefix
	// and are exactly 3 bytes long.
	for _, m := range [][]byte{StartBytes, EndBytes, HashPrefixBytes} {
		if len(m) != markerLen || m[0] != StartBytes[0] || m[1] != StartBytes[1] {
			panic("marker characters must be 3-byte UTF-8 with shared prefix")
		}
	}
}

// stripMarkersBytes scans data for marker characters (‹, ›, †) and either
// removes them (when replacement is nil) or replaces them with the
// replacement bytes.
func stripMarkersBytes(data []byte, replacement []byte) []byte {
	lead := StartBytes[0] // first byte shared by all marker chars
	// Fast path: no marker characters possible.
	first := bytes.IndexByte(data, lead)
	if first == -1 {
		return data
	}

	mid := StartBytes[1] // second byte shared by all marker chars
	b2Start := StartBytes[2]
	b2End := EndBytes[2]
	b2Hash := HashPrefixBytes[2]

	buf := make([]byte, 0, len(data))
	pos := 0
	for i := first; i < len(data); {
		if data[i] == lead && i+2 < len(data) && data[i+1] == mid {
			if b := data[i+2]; b == b2Start || b == b2End || b == b2Hash {
				buf = append(buf, data[pos:i]...)
				buf = append(buf, replacement...)
				i += markerLen
				pos = i
				continue
			}
		}
		i++
	}
	if pos == 0 {
		return data
	}
	buf = append(buf, data[pos:]...)
	return buf
}
