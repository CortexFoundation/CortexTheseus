// Copyright 2018 The go-ethereum Authors
// This file is part of CortexFoundation.
//
// CortexFoundation is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// CortexFoundation is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with CortexFoundation. If not, see <http://www.gnu.org/licenses/>.

package main

import (
	"fmt"
	"os"
	"runtime"
	"strings"

	"gopkg.in/urfave/cli.v1"

	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/ctxc"

	// "github.com/CortexFoundation/CortexTheseus/consensus/cuckoo"
	"github.com/CortexFoundation/CortexTheseus/params"
)

var (
	/* makecacheCommand = cli.Command{
			Action:    utils.MigrateFlags(makecache),
			Name:      "makecache",
			Usage:     "Generate ctxcash verification cache (for testing)",
			ArgsUsage: "<blockNum> <outputDir>",
			Category:  "MISCELLANEOUS COMMANDS",
			Description: `
	The makecache command generates an ctxcash cache in <outputDir>.

	This command exists to support the system testing project.
	Regular users do not need to execute it.
	`,
		} */
	/* makedagCommand = cli.Command{
			Action:    utils.MigrateFlags(makedag),
			Name:      "makedag",
			Usage:     "Generate ctxcash mining DAG (for testing)",
			ArgsUsage: "<blockNum> <outputDir>",
			Category:  "MISCELLANEOUS COMMANDS",
			Description: `
	The makedag command generates an ctxcash DAG in <outputDir>.

	This command exists to support the system testing project.
	Regular users do not need to execute it.
	`,
		} */
	versionCommand = cli.Command{
		Action:    utils.MigrateFlags(version),
		Name:      "version",
		Usage:     "Print version numbers",
		ArgsUsage: " ",
		Category:  "MISCELLANEOUS COMMANDS",
		Description: `
The output of this command is supposed to be machine-readable.
`,
	}
	licenseCommand = cli.Command{
		Action:    utils.MigrateFlags(license),
		Name:      "license",
		Usage:     "Display license information",
		ArgsUsage: " ",
		Category:  "MISCELLANEOUS COMMANDS",
	}
)

// makecache generates an ctxcash verification cache into the provided folder.
/* func makecache(ctx *cli.Context) error {
	args := ctx.Args()
	if len(args) != 2 {
		utils.Fatalf(`Usage: cortex makecache <block number> <outputdir>`)
	}
	block, err := strconv.ParseUint(args[0], 0, 64)
	if err != nil {
		utils.Fatalf("Invalid block number: %v", err)
	}
	ctxcash.MakeCache(block, args[1])

	return nil
} */

// makedag generates an ctxcash mining DAG into the provided folder.
/* func makedag(ctx *cli.Context) error {
	args := ctx.Args()
	if len(args) != 2 {
		utils.Fatalf(`Usage: cortex makedag <block number> <outputdir>`)
	}
	block, err := strconv.ParseUint(args[0], 0, 64)
	if err != nil {
		utils.Fatalf("Invalid block number: %v", err)
	}
	ctxcash.MakeDataset(block, args[1])

	return nil
} */

func version(ctx *cli.Context) error {
	fmt.Println(strings.Title(clientIdentifier))
	fmt.Println("Version:", params.VersionWithMeta)
	if gitCommit != "" {
		fmt.Println("Git Commit:", gitCommit)
	}
	fmt.Println("Architecture:", runtime.GOARCH)
	fmt.Println("Protocol Versions:", ctxc.ProtocolVersions)
	fmt.Println("Network Id:", ctxc.DefaultConfig.NetworkId)
	fmt.Println("Go Version:", runtime.Version())
	fmt.Println("Operating System:", runtime.GOOS)
	fmt.Printf("GOPATH=%s\n", os.Getenv("GOPATH"))
	fmt.Printf("GOROOT=%s\n", runtime.GOROOT())
	return nil
}

func license(_ *cli.Context) error {
	fmt.Println(`Ctxc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Ctxc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with cortex. If not, see <http://www.gnu.org/licenses/>.`)
	return nil
}
