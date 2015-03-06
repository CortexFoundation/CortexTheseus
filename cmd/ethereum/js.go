// Copyright (c) 2013-2014, Jeffrey Wilcke. All rights reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
// MA 02110-1301  USA

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strings"

	"github.com/ethereum/go-ethereum/cmd/utils"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/eth"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/javascript"
	"github.com/ethereum/go-ethereum/state"
	"github.com/ethereum/go-ethereum/xeth"
	"github.com/obscuren/otto"
	"github.com/peterh/liner"
)

func execJsFile(ethereum *eth.Ethereum, filename string) {
	file, err := os.Open(filename)
	if err != nil {
		utils.Fatalf("%v", err)
	}
	content, err := ioutil.ReadAll(file)
	if err != nil {
		utils.Fatalf("%v", err)
	}
	re := javascript.NewJSRE(xeth.New(ethereum))
	if _, err := re.Run(string(content)); err != nil {
		utils.Fatalf("Javascript Error: %v", err)
	}
}

type repl struct {
	re       *javascript.JSRE
	ethereum *eth.Ethereum
	xeth     *xeth.XEth
	prompt   string
	histfile *os.File
	lr       *liner.State
	running  bool
}

func newREPL(ethereum *eth.Ethereum) *repl {
	hist, err := os.OpenFile(path.Join(ethereum.DataDir, "history"), os.O_RDWR|os.O_CREATE, os.ModePerm)
	if err != nil {
		panic(err)
	}
	xeth := xeth.New(ethereum)
	repl := &repl{
		re:       javascript.NewJSRE(xeth),
		xeth:     xeth,
		ethereum: ethereum,
		prompt:   "> ",
		histfile: hist,
		lr:       liner.NewLiner(),
	}
	repl.initStdFuncs()
	return repl
}

func (self *repl) Start() {
	if !self.running {
		self.running = true
		self.lr.ReadHistory(self.histfile)
		go self.read()
	}
}

func (self *repl) Stop() {
	if self.running {
		self.running = false
		self.histfile.Truncate(0)
		self.lr.WriteHistory(self.histfile)
		self.histfile.Close()
	}
}

func (self *repl) parseInput(code string) {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("[native] error", r)
		}
	}()
	value, err := self.re.Run(code)
	if err != nil {
		fmt.Println(err)
		return
	}
	self.printValue(value)
}

var indentCount = 0
var str = ""

func (self *repl) setIndent() {
	open := strings.Count(str, "{")
	open += strings.Count(str, "(")
	closed := strings.Count(str, "}")
	closed += strings.Count(str, ")")
	indentCount = open - closed
	if indentCount <= 0 {
		self.prompt = "> "
	} else {
		self.prompt = strings.Join(make([]string, indentCount*2), "..")
		self.prompt += " "
	}
}

func (self *repl) read() {
	for {
		input, err := self.lr.Prompt(self.prompt)
		if err != nil {
			return
		}
		if input == "" {
			continue
		}
		str += input + "\n"
		self.setIndent()
		if indentCount <= 0 {
			if input == "exit" {
				self.Stop()
				return
			}
			hist := str[:len(str)-1]
			self.lr.AppendHistory(hist)
			self.parseInput(str)
			str = ""
		}
	}
}

func (self *repl) printValue(v interface{}) {
	method, _ := self.re.Vm.Get("prettyPrint")
	v, err := self.re.Vm.ToValue(v)
	if err == nil {
		val, err := method.Call(method, v)
		if err == nil {
			fmt.Printf("%v", val)
		}
	}
}

func (self *repl) initStdFuncs() {
	t, _ := self.re.Vm.Get("eth")
	eth := t.Object()
	eth.Set("connect", self.connect)
	eth.Set("stopMining", self.stopMining)
	eth.Set("startMining", self.startMining)
	eth.Set("dump", self.dump)
	eth.Set("export", self.export)
}

/*
 * The following methods are natively implemented javascript functions.
 */

func (self *repl) dump(call otto.FunctionCall) otto.Value {
	var block *types.Block

	if len(call.ArgumentList) > 0 {
		if call.Argument(0).IsNumber() {
			num, _ := call.Argument(0).ToInteger()
			block = self.ethereum.ChainManager().GetBlockByNumber(uint64(num))
		} else if call.Argument(0).IsString() {
			hash, _ := call.Argument(0).ToString()
			block = self.ethereum.ChainManager().GetBlock(ethutil.Hex2Bytes(hash))
		} else {
			fmt.Println("invalid argument for dump. Either hex string or number")
		}

		if block == nil {
			fmt.Println("block not found")

			return otto.UndefinedValue()
		}

	} else {
		block = self.ethereum.ChainManager().CurrentBlock()
	}

	statedb := state.New(block.Root(), self.ethereum.Db())

	v, _ := self.re.Vm.ToValue(statedb.RawDump())

	return v
}

func (self *repl) stopMining(call otto.FunctionCall) otto.Value {
	self.xeth.Miner().Stop()
	return otto.TrueValue()
}

func (self *repl) startMining(call otto.FunctionCall) otto.Value {
	self.xeth.Miner().Start()
	return otto.TrueValue()
}

func (self *repl) connect(call otto.FunctionCall) otto.Value {
	nodeURL, err := call.Argument(0).ToString()
	if err != nil {
		return otto.FalseValue()
	}
	if err := self.ethereum.SuggestPeer(nodeURL); err != nil {
		return otto.FalseValue()
	}
	return otto.TrueValue()
}

func (self *repl) export(call otto.FunctionCall) otto.Value {
	if len(call.ArgumentList) == 0 {
		fmt.Println("err: require file name")
		return otto.FalseValue()
	}

	fn, err := call.Argument(0).ToString()
	if err != nil {
		fmt.Println(err)
		return otto.FalseValue()
	}

	data := self.ethereum.ChainManager().Export()

	if err := ethutil.WriteFile(fn, data); err != nil {
		fmt.Println(err)
		return otto.FalseValue()
	}

	return otto.TrueValue()
}
