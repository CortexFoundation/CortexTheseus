/*
	This file is part of go-ethereum

	go-ethereum is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	go-ethereum is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with go-ethereum.  If not, see <http://www.gnu.org/licenses/>.
*/
/**
 * @authors
 * 	Jeffrey Wilcke <i@jev.io>
 */
package main

import (
	"encoding/json"
	"os"
	"strconv"

	"github.com/ethereum/go-ethereum/cmd/utils"
	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/ethutil"
	"github.com/ethereum/go-ethereum/logger"
	"github.com/ethereum/go-ethereum/state"
)

type plugin struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

// LogPrint writes to the GUI log.
func (gui *Gui) LogPrint(level logger.LogLevel, msg string) {
	/*
		str := strings.TrimRight(s, "\n")
		lines := strings.Split(str, "\n")

		view := gui.getObjectByName("infoView")
		for _, line := range lines {
			view.Call("addLog", line)
		}
	*/
}
func (gui *Gui) Transact(recipient, value, gas, gasPrice, d string) (string, error) {
	var data string
	if len(recipient) == 0 {
		code, err := ethutil.Compile(d, false)
		if err != nil {
			return "", err
		}
		data = ethutil.Bytes2Hex(code)
	} else {
		data = ethutil.Bytes2Hex(utils.FormatTransactionData(d))
	}

	return gui.xeth.Transact(recipient, value, gas, gasPrice, data)
}

// functions that allow Gui to implement interface guilogger.LogSystem
func (gui *Gui) SetLogLevel(level logger.LogLevel) {
	gui.logLevel = level
	gui.eth.Logger().SetLogLevel(level)
	gui.config.Save("loglevel", level)
}

func (gui *Gui) GetLogLevel() logger.LogLevel {
	return gui.logLevel
}

func (self *Gui) AddPlugin(pluginPath string) {
	self.plugins[pluginPath] = plugin{Name: pluginPath, Path: pluginPath}

	json, _ := json.MarshalIndent(self.plugins, "", "    ")
	ethutil.WriteFile(self.eth.DataDir+"/plugins.json", json)
}

func (self *Gui) RemovePlugin(pluginPath string) {
	delete(self.plugins, pluginPath)

	json, _ := json.MarshalIndent(self.plugins, "", "    ")
	ethutil.WriteFile(self.eth.DataDir+"/plugins.json", json)
}

// this extra function needed to give int typecast value to gui widget
// that sets initial loglevel to default
func (gui *Gui) GetLogLevelInt() int {
	return int(gui.logLevel)
}
func (self *Gui) DumpState(hash, path string) {
	var stateDump []byte

	if len(hash) == 0 {
		stateDump = self.eth.ChainManager().State().Dump()
	} else {
		var block *types.Block
		if hash[0] == '#' {
			i, _ := strconv.Atoi(hash[1:])
			block = self.eth.ChainManager().GetBlockByNumber(uint64(i))
		} else {
			block = self.eth.ChainManager().GetBlock(ethutil.Hex2Bytes(hash))
		}

		if block == nil {
			guilogger.Infof("block err: not found %s\n", hash)
			return
		}

		stateDump = state.New(block.Root(), self.eth.Db()).Dump()
	}

	file, err := os.OpenFile(path[7:], os.O_CREATE|os.O_RDWR, os.ModePerm)
	if err != nil {
		guilogger.Infoln("dump err: ", err)
		return
	}
	defer file.Close()

	guilogger.Infof("dumped state (%s) to %s\n", hash, path)

	file.Write(stateDump)
}
