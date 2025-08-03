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

import (
	"bytes"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/CortexFoundation/CortexTheseus/log"
)

func (wh *Wormhole) healthCheck(s string) error {
	log.Debug("Global best trackers", "url", s)

	switch {
	case strings.HasPrefix(s, "http"), strings.HasPrefix(s, "https"):
		if err := checkHTTPTracker(s); err != nil {
			log.Debug("tracker failed", "err", err)
			return err
		}
		return nil

	case strings.HasPrefix(s, "udp"):
		u, err := url.Parse(s)
		if err != nil {
			return err
		}
		host, port, err := net.SplitHostPort(u.Host)
		if err != nil {
			return err
		}
		if err := checkUDPTracker(net.JoinHostPort(host, port)); err != nil {
			log.Debug("UDP ping err", "s", s, "err", err)
			return err
		}
		return nil

	default:
		log.Warn("Other protocols trackers", "s", s)
		return errors.New("invalid url protocol")
	}
}

func random20Bytes() []byte {
	b := make([]byte, 20)
	rand.Read(b)
	return b
}

func buildAnnounceURL(base string) (string, error) {
	parsed, err := url.Parse(base)
	if err != nil {
		return "", err
	}

	infoHash := random20Bytes()
	peerID := []byte("-GT0001-" + hex.EncodeToString(random20Bytes())[:12])
	values := url.Values{
		"info_hash":  {string(infoHash)},
		"peer_id":    {string(peerID)},
		"port":       {"6881"},
		"uploaded":   {"0"},
		"downloaded": {"0"},
		"left":       {"0"},
		"compact":    {"1"},
		"event":      {"started"},
	}

	parsed.RawQuery = values.Encode()
	return parsed.String(), nil
}

func checkHTTPTracker(base string) error {
	fullURL, err := buildAnnounceURL(base)
	if err != nil {
		return fmt.Errorf("invalid tracker URL: %w", err)
	}

	client := http.Client{
		Timeout: 5 * time.Second,
	}

	resp, err := client.Get(fullURL)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected HTTP status: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read tracker response: %w", err)
	}

	if bytes.Contains(body, []byte("failure reason")) {
		return fmt.Errorf("tracker responded with failure")
	}

	return nil
}

const (
	udpTimeout    = 5 * time.Second
	actionConnect = 0
	protocolID    = 0x41727101980
)

func checkUDPTracker(trackerHostPort string) error {
	host, port, err := net.SplitHostPort(trackerHostPort)
	if err != nil {
		return fmt.Errorf("invalid tracker URL: %w", err)
	}

	addr, err := net.ResolveUDPAddr("udp", net.JoinHostPort(host, port))
	if err != nil {
		return fmt.Errorf("resolve failed: %w", err)
	}

	conn, err := net.DialUDP("udp", nil, addr)
	if err != nil {
		return fmt.Errorf("dial failed: %w", err)
	}
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(udpTimeout))

	transactionID := make([]byte, 4)
	rand.Read(transactionID)

	var buf bytes.Buffer
	binary.Write(&buf, binary.BigEndian, uint64(protocolID))
	binary.Write(&buf, binary.BigEndian, uint32(actionConnect))
	buf.Write(transactionID)

	if _, err = conn.Write(buf.Bytes()); err != nil {
		return fmt.Errorf("send connect request failed: %w", err)
	}

	resp := make([]byte, 16)
	if _, err = conn.Read(resp); err != nil {
		return fmt.Errorf("connect response failed: %w", err)
	}

	if len(resp) < 16 || binary.BigEndian.Uint32(resp[:4]) != actionConnect || !bytes.Equal(resp[4:8], transactionID) {
		return fmt.Errorf("invalid connect response")
	}

	return nil
}
