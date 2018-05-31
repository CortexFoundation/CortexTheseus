// Copyright 2017 The go-ethereum Authors
// This file is part of go-ethereum.
//
// go-ethereum is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// go-ethereum is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with go-ethereum. If not, see <http://www.gnu.org/licenses/>.

package main

import (
	"encoding/hex"
	"encoding/json"
	"github.com/ethereum/go-ethereum/common"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/rlp"
	cli "gopkg.in/urfave/cli.v1"

	"fmt"
)

var extCommand = cli.Command{
	Action:      extCmd,
	Name:        "ext",
	Usage:       "tool to encode model meta and input meta",
	ArgsUsage:   "<code>",
	Description: `tool to encode model meta and input meta`,
}

type ModelMetaExt struct {
	Hash          string
	RawSize       uint64
	InputShape    []uint64
	OutputShape   []uint64
	Gas           uint64
	AuthorAddress string `json:"author"`
}

type InputMetaExt struct {
	Hash          string
	RawSize       uint64
	Shape         []uint64
	AuthorAddress string
}

func extCmd(ctx *cli.Context) error {
	if ctx.GlobalString(MetaJsonFlag.Name) != "" {
		fmt.Println(ctx.GlobalBool(ParseModelMetaFlag.Name))
		json_data := []byte(ctx.GlobalString(MetaJsonFlag.Name))
		fmt.Println(string(json_data))
		if ctx.GlobalBool(ParseModelMetaFlag.Name) {

			var model ModelMetaExt
			json.Unmarshal(json_data, &model)

			out, _ := rlp.EncodeToBytes(
				&types.ModelMeta{
					Hash:          []byte(model.Hash),
					RawSize:       model.RawSize,
					InputShape:    model.InputShape,
					OutputShape:   model.OutputShape,
					Gas:           model.Gas,
					AuthorAddress: common.BytesToAddress([]byte(model.AuthorAddress)),
				})
			fmt.Println(model)
			fmt.Println(hex.EncodeToString(append([]byte{0, 1}, out...)))
		} else {

			var input InputMetaExt
			json.Unmarshal(json_data, &input)

			out, _ := rlp.EncodeToBytes(
				&types.InputMeta{
					Hash:          []byte(input.Hash),
					RawSize:       input.RawSize,
					Shape:         input.Shape,
					AuthorAddress: common.BytesToAddress([]byte(input.AuthorAddress)),
				})
			fmt.Println(out)
		}
	}
	return nil
}
