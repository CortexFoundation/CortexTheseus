// Copyright 2018 The CortexTheseus Authors
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
	"crypto/ecdsa"
	"fmt"
	//"io/ioutil"
	//"os"
	//"path/filepath"

	"github.com/CortexFoundation/CortexTheseus/cmd/utils"
	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/crypto"
	"gopkg.in/urfave/cli.v1"
)

var commandBatchGenerate = cli.Command{
	Name:      "batch_generate",
	Usage:     "batch generate new pk",
	ArgsUsage: "[ <keyfile> ]",
	Description: `
Generate a new keyfile.

If you want to encrypt an existing private key, it can be specified by setting
--privatekey with the location of the file containing the private key.
`,
	Flags: []cli.Flag{
		batchFlag,
	},
	Action: func(ctx *cli.Context) error {
		// Check if keyfile path given and make sure it doesn't already exist.
		/*keyfilepath := ctx.Args().First()
		if keyfilepath == "" {
			keyfilepath = "batch_keyfile.json" //defaultKeyfileName
		}
		if _, err := os.Stat(keyfilepath); err == nil {
			utils.Fatalf("Keyfile already exists at %s.", keyfilepath)
		} else if !os.IsNotExist(err) {
			utils.Fatalf("Error checking if keyfile exists: %v", err)
		}*/

		var privateKey *ecdsa.PrivateKey
		var err error
		for i := 0; i < ctx.Int("batch"); i++ {
			privateKey, err = crypto.GenerateKey()
			if err != nil {
				utils.Fatalf("Failed to generate random private key: %v", err)
			}

			k := common.ToHex(crypto.FromECDSA(privateKey))
			address := crypto.PubkeyToAddress(privateKey.PublicKey)
			fmt.Println(k)
			fmt.Println(address.Hex())
			//			p := common.ToHex(crypto.FromECDSAPub(&privateKey.PublicKey))
			//			fmt.Println(p)

			// Store the file to disk.
			/*if err := os.MkdirAll(filepath.Dir(keyfilepath), 0700); err != nil {
				utils.Fatalf("Could not create directory %s", filepath.Dir(keyfilepath))
			}
			//keyjson:= " " + common.ToHex(crypto.FromECDSAPub(&privateKey.PublicKey))
			keyjson:= crypto.FromECDSAPub(&privateKey.PublicKey)
			if err := ioutil.WriteFile(keyfilepath, keyjson, 0600); err != nil {
				utils.Fatalf("Failed to write keyfile to %s: %v", keyfilepath, err)
			}*/
		}

		// Output some information.
		/*out := outputGenerate{
			Address: key.Address.Hex(),
		}
		if ctx.Bool(jsonFlag.Name) {
			mustPrintJSON(out)
		} else {
			fmt.Println("Address:", out.Address)
		}*/
		return nil
	},
}
