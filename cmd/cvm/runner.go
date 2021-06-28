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

	"gopkg.in/urfave/cli.v1"
)

var runCommand = cli.Command{
	Action:      runCmd,
	Name:        "run",
	Usage:       "run arbitrary cvm binary",
	ArgsUsage:   "<code>",
	Description: `The run command runs arbitrary CVM code.`,
}

func runCmd(ctx *cli.Context) error {
	fmt.Printf("Runner Started!")
	return nil
}
