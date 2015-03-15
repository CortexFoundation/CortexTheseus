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
	"bufio"
	"fmt"
	"os"
	"path"
	"strings"

	"github.com/ethereum/go-ethereum/cmd/utils"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/eth"
	re "github.com/ethereum/go-ethereum/jsre"
	"github.com/ethereum/go-ethereum/rpc"
	"github.com/ethereum/go-ethereum/xeth"
	"github.com/peterh/liner"
)

type prompter interface {
	AppendHistory(string)
	Prompt(p string) (string, error)
	PasswordPrompt(p string) (string, error)
}

type dumbterm struct{ r *bufio.Reader }

func (r dumbterm) Prompt(p string) (string, error) {
	fmt.Print(p)
	return r.r.ReadString('\n')
}

func (r dumbterm) PasswordPrompt(p string) (string, error) {
	fmt.Println("!! Unsupported terminal, password will echo.")
	fmt.Print(p)
	input, err := bufio.NewReader(os.Stdin).ReadString('\n')
	fmt.Println()
	return input, err
}

func (r dumbterm) AppendHistory(string) {}

type jsre struct {
	re       *re.JSRE
	ethereum *eth.Ethereum
	xeth     *xeth.XEth
	ps1      string
	atexit   func()

	prompter
}

func newJSRE(ethereum *eth.Ethereum, libPath string) *jsre {
	js := &jsre{ethereum: ethereum, ps1: "> "}
	js.xeth = xeth.New(ethereum, js)
	js.re = re.New(libPath)
	js.apiBindings()
	js.adminBindings()

	if !liner.TerminalSupported() {
		js.prompter = dumbterm{bufio.NewReader(os.Stdin)}
	} else {
		lr := liner.NewLiner()
		js.withHistory(func(hist *os.File) { lr.ReadHistory(hist) })
		lr.SetCtrlCAborts(true)
		js.prompter = lr
		js.atexit = func() {
			js.withHistory(func(hist *os.File) { hist.Truncate(0); lr.WriteHistory(hist) })
			lr.Close()
		}
	}
	return js
}

func (js *jsre) apiBindings() {

	ethApi := rpc.NewEthereumApi(js.xeth, js.ethereum.DataDir)
	js.re.Bind("jeth", rpc.NewJeth(ethApi, js.re.ToVal))

	_, err := js.re.Eval(re.BigNumber_JS)

	if err != nil {
		utils.Fatalf("Error loading bignumber.js: %v", err)
	}

	// we need to declare a dummy setTimeout. Otto does not support it
	_, err = js.re.Eval("setTimeout = function(cb, delay) {};")
	if err != nil {
		utils.Fatalf("Error defining setTimeout: %v", err)
	}

	_, err = js.re.Eval(re.Ethereum_JS)
	if err != nil {
		utils.Fatalf("Error loading ethereum.js: %v", err)
	}

	_, err = js.re.Eval("var web3 = require('web3');")
	if err != nil {
		utils.Fatalf("Error requiring web3: %v", err)
	}

	_, err = js.re.Eval("web3.setProvider(jeth)")
	if err != nil {
		utils.Fatalf("Error setting web3 provider: %v", err)
	}
	_, err = js.re.Eval(`
	var eth = web3.eth;
  var shh = web3.shh;
  var db  = web3.db;
  var net = web3.net;
  `)
	if err != nil {
		utils.Fatalf("Error setting namespaces: %v", err)
	}

}

func (self *jsre) ConfirmTransaction(tx *types.Transaction) bool {
	p := fmt.Sprintf("Confirm Transaction %v\n[y/n] ", tx)
	answer, _ := self.Prompt(p)
	return strings.HasPrefix(strings.Trim(answer, " "), "y")
}

func (self *jsre) UnlockAccount(addr []byte) bool {
	fmt.Printf("Please unlock account %x.\n", addr)
	pass, err := self.PasswordPrompt("Passphrase: ")
	if err != nil {
		return false
	}
	// TODO: allow retry
	if err := self.ethereum.AccountManager().Unlock(addr, pass); err != nil {
		return false
	} else {
		fmt.Println("Account is now unlocked for this session.")
		return true
	}
}

func (self *jsre) exec(filename string) error {
	if err := self.re.Exec(filename); err != nil {
		return fmt.Errorf("Javascript Error: %v", err)
	}
	return nil
}

func (self *jsre) interactive() {
	for {
		input, err := self.Prompt(self.ps1)
		if err != nil {
			break
		}
		if input == "" {
			continue
		}
		str += input + "\n"
		self.setIndent()
		if indentCount <= 0 {
			if input == "exit" {
				break
			}
			hist := str[:len(str)-1]
			self.AppendHistory(hist)
			self.parseInput(str)
			str = ""
		}
	}
	if self.atexit != nil {
		self.atexit()
	}
}

func (self *jsre) withHistory(op func(*os.File)) {
	hist, err := os.OpenFile(path.Join(self.ethereum.DataDir, "history"), os.O_RDWR|os.O_CREATE, os.ModePerm)
	if err != nil {
		fmt.Printf("unable to open history file: %v\n", err)
		return
	}
	op(hist)
	hist.Close()
}

func (self *jsre) parseInput(code string) {
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

func (self *jsre) setIndent() {
	open := strings.Count(str, "{")
	open += strings.Count(str, "(")
	closed := strings.Count(str, "}")
	closed += strings.Count(str, ")")
	indentCount = open - closed
	if indentCount <= 0 {
		self.ps1 = "> "
	} else {
		self.ps1 = strings.Join(make([]string, indentCount*2), "..")
		self.ps1 += " "
	}
}

func (self *jsre) printValue(v interface{}) {
	val, err := self.re.PrettyPrint(v)
	if err == nil {
		fmt.Printf("%v", val)
	}
}
