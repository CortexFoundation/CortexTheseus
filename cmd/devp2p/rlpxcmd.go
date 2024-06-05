// Copyright 2020 The go-ethereum Authors
// This file is part of CortexTheseus.
//
// CortexTheseus is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CortexTheseus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CortexTheseus. If not, see <http://www.gnu.org/licenses/>.

package main

import (
	"fmt"
	"net"

	"gopkg.in/urfave/cli.v1"

	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/p2p/rlpx"
	"github.com/CortexFoundation/CortexTheseus/rlp"
)

var (
	rlpxCommand = cli.Command{
		Name:  "rlpx",
		Usage: "RLPx Commands",
		Subcommands: []cli.Command{
			rlpxPingCommand,
		},
	}
	rlpxPingCommand = cli.Command{
		Name:      "ping",
		Usage:     "Perform a RLPx handshake",
		ArgsUsage: "<node>",
		Action:    rlpxPing,
	}
)

func rlpxPing(ctx *cli.Context) error {
	n := getNodeArg(ctx)
	tcpEndpoint, ok := n.TCPEndpoint()
	if !ok {
		return fmt.Errorf("node has no TCP endpoint")
	}
	fd, err := net.Dial("tcp", tcpEndpoint.String())
	if err != nil {
		return err
	}
	conn := rlpx.NewConn(fd, n.Pubkey())

	ourKey, _ := crypto.GenerateKey()
	_, err = conn.Handshake(ourKey)
	if err != nil {
		return err
	}

	code, data, _, err := conn.Read()
	if err != nil {
		return err
	}
	switch code {
	case 0:
		var h devp2pHandshake
		if err := rlp.DecodeBytes(data, &h); err != nil {
			return fmt.Errorf("invalid handshake: %v", err)
		}
		fmt.Printf("%+v\n", h)
	case 1:
		var msg []p2p.DiscReason
		if rlp.DecodeBytes(data, &msg); len(msg) == 0 {
			return fmt.Errorf("invalid disconnect message")
		}
		return fmt.Errorf("received disconnect message: %v", msg[0])
	default:
		return fmt.Errorf("invalid message code %d, expected handshake (code zero)", code)
	}
	return nil
}

// devp2pHandshake is the RLP structure of the devp2p protocol handshake.
type devp2pHandshake struct {
	Version    uint64
	Name       string
	Caps       []p2p.Cap
	ListenPort uint64
	ID         hexutil.Bytes // secp256k1 public key
	// Ignore additional fields (for forward compatibility).
	Rest []rlp.RawValue `rlp:"tail"`
}
