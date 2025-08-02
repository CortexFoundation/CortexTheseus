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
		//if _, err := wh.cl.R().Post(s); err != nil {
		if err := checkHTTPTracker(s); err != nil {
			log.Debug("tracker failed", "err", err)
			// TODO
			return err
		} else {
			//ret = append(ret, s)
			return nil
		}
	case strings.HasPrefix(s, "udp"):
		if u, err := url.Parse(s); err == nil {
			if host, port, err := net.SplitHostPort(u.Host); err == nil {
				if err := ping(host, port); err == nil {
					//ret = append(ret, s)
					return nil
				} else {
					log.Debug("UDP ping err", "s", s, "err", err)
					// TODO
					return err
				}
			}
		} else {
			return err
		}
	default:
		log.Warn("Other protocols trackers", "s", s)
		return errors.New("invalid url protocol")
	}

	return errors.New("unhealthy tracker url")
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

	//fmt.Printf("Tracker responded (%d bytes)\n", len(body))
	return nil
}

func ping(host, port string) error {
	address := net.JoinHostPort(host, port)

	return checkUDPTracker(address)

	/*raddr, err := net.ResolveUDPAddr("udp", address)
	if err != nil {
		return fmt.Errorf("failed to resolve address: %w", err)
	}

	conn, err := net.DialUDP("udp", nil, raddr)
	if err != nil {
		return fmt.Errorf("failed to dial UDP: %w", err)
	}
	defer conn.Close()

	_, err = conn.Write([]byte{})
	if err != nil {
		return fmt.Errorf("failed to send UDP packet: %w", err)
	}

	return nil*/
}

/*
	func ping(host string, port string) error {
		address := net.JoinHostPort(host, port)
		raddr, err1 := net.ResolveUDPAddr("udp", address)
		if err1 != nil {
			return err1
		}
		conn, err := net.DialUDP("udp", nil, raddr)
		if conn != nil {
			defer conn.Close()
		}
		return err
	}
*/
const (
	udpTimeout     = 5 * time.Second
	actionConnect  = 0
	actionAnnounce = 1
	protocolID     = 0x41727101980
)

func checkUDPTracker(trackerURL string) error {
	host, port, err := net.SplitHostPort(trackerURL[6:])
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
	binary.Write(&buf, binary.BigEndian, uint64(protocolID))    // protocol ID
	binary.Write(&buf, binary.BigEndian, uint32(actionConnect)) // action = connect
	buf.Write(transactionID)                                    // transaction ID

	_, err = conn.Write(buf.Bytes())
	if err != nil {
		return fmt.Errorf("send connect request failed: %w", err)
	}

	resp := make([]byte, 16)
	_, err = conn.Read(resp)
	if err != nil {
		return fmt.Errorf("connect response failed: %w", err)
	}

	if len(resp) < 16 || resp[0] != 0 || !bytes.Equal(resp[4:8], transactionID) {
		return fmt.Errorf("invalid connect response")
	}

	//connectionID := resp[8:16]

	//fmt.Println("Tracker responded to connect request.")
	//fmt.Printf("Connection ID: %x\n", connectionID)
	return nil
}
