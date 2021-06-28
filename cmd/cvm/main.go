// Copyright 2021 The CortexTheseus Authors
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
	"os"

	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/log"
	"gopkg.in/urfave/cli.v1"
)

var (
	// Git SHA1 commit hash of the release (set via linker flags)
	gitCommit = ""
	gitDate   = ""

	app *cli.App

	CodeFlag = cli.StringFlag{
		Name:  "code",
		Usage: "CVM code",
	}
	CodeFileFlag = cli.StringFlag{
		Name:  "codefile",
		Usage: "File containing CVM code. If '-' is specified, code is read from stdin ",
	}
)

func init() {
	app = utils.NewApp(gitCommit, "CortexFoundation checkpoint helper tool")
	app.Flags = []cli.Flag{
		CodeFlag,
		CodeFileFlag,
	}
	app.Commands = []cli.Command{
		runCommand,
	}

	cli.CommandHelpTemplate = utils.OriginCommandHelpTemplate
}

func main() {
	log.Root().SetHandler(log.LvlFilterHandler(log.LvlInfo, log.StreamHandler(os.Stderr, log.TerminalFormat(true))))

	if err := app.Run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
