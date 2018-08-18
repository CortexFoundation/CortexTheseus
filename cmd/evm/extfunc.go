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
	AuthorAddress string
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

			var model_tmp ModelMetaExt
			json.Unmarshal(json_data, &model_tmp)
			model := &types.ModelMeta{
				Hash:          common.HexToAddress(model_tmp.Hash),
				RawSize:       model_tmp.RawSize,
				InputShape:    model_tmp.InputShape,
				OutputShape:   model_tmp.OutputShape,
				Gas:           model_tmp.Gas,
				AuthorAddress: common.BytesToAddress([]byte(model_tmp.AuthorAddress)),
			}
			out, _ := rlp.EncodeToBytes(model)
			fmt.Printf("payload: %v\n", hex.EncodeToString(append([]byte{0, 1}, out...)))
		} else {
			var data_tmp InputMetaExt
			json.Unmarshal(json_data, &data_tmp)
			data := &types.InputMeta{
				Hash:          common.HexToAddress(data_tmp.Hash),
				RawSize:       data_tmp.RawSize,
				Shape:         data_tmp.Shape,
				AuthorAddress: common.BytesToAddress([]byte(data_tmp.AuthorAddress)),
			}
			out, _ := rlp.EncodeToBytes(data)
			fmt.Printf("payload: %v\n", hex.EncodeToString(append([]byte{0, 2}, out...)))
		}
	}
	return nil
}
