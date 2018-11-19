package main

import (
	"crypto/ecdsa"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"path/filepath"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/p2p/discover"
)

var (
	keyGenerate      = flag.Bool("keygen", false, "Generate nodekey")
	keyBatchGenerate = flag.Bool("batchgen", false, "Batch generate nodekey")
	keysLookup       = flag.Bool("lookup", false, "Look up keys in directory")

	keysDir = flag.String("dir", ".", "Generate or lookup directory")
	number  = flag.Int("number", 10, "Batch generate keys number")
)

func main() {
	flag.Parse()

	switch {
	case (*keyBatchGenerate):
		BatchKeyGenerate()
		break
	case (*keyGenerate):
		KeyGenerate()
		break
	case (*keysLookup):
		KeysLookup()
		break
	default:
		PrintHelp()
		break
	}
}

func PrintHelp() {
	log.Println(fmt.Sprintf(
		`
Usage:	nodekey -[command] --[options]

COMMAND:
	keygen:		Generate private key for node and print public key
	batchgen:	Batch generate private key
	lookup:		Lookup private key's public key

OPTIONS:
	dir:		The directory for key generate or lookup, invalid keyfile 
				will be ignored. default is current directory
	number:		Number of keys to batch generate. Default value is 10
`))
}

func BatchKeyGenerate() {
	for i := 0; i < (*number); i++ {
		name := fmt.Sprintf("nodekey_%d", i)
		keyfile := filepath.Join(*keysDir, name)
		keyGen(keyfile)
	}
}

func KeyGenerate() {
	keyfile := filepath.Join(*keysDir, "nodekey")
	keyGen(keyfile)
}

func keyGen(keyfile string) {
	key, err := crypto.GenerateKey()
	if err != nil {
		log.Fatalln(fmt.Sprintf("Failed to generate node key: %v", err))
		return
	}

	fmt.Println(keyfile + "\tpublic key: " + discover.PubkeyID(&key.PublicKey).String())

	if err := crypto.SaveECDSA(keyfile, key); err != nil {
		log.Fatalln(fmt.Sprintf("Failed to persist node key: %v", err))
		return
	}
}

func KeysLookup() {
	files, err := ioutil.ReadDir(*keysDir)
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range files {
		ExportKeys(filepath.Join((*keysDir), file.Name()))
		// if err := ExportKeys((*keysDir) + file.Name()); err != nil {
		// 	fmt.Println("Export keys invalid, error: " + err.Error())
		// }
	}
}

// type WalkFunc func(path string, info os.FileInfo, err error) error
func ExportKeys(file string) error {
	var (
		key *ecdsa.PrivateKey
		err error
	)

	if key, err = crypto.LoadECDSA(file); err != nil {
		return errors.New(fmt.Sprintf("Load ECDSA from file error | %v", err))
	}

	fmt.Println(file + "\t loaded")
	fmt.Println("\t" + discover.PubkeyID(&key.PublicKey).String())
	return nil
}
