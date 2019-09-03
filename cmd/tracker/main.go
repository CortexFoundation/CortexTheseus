package main

import (
	"fmt"
	"github.com/CortexFoundation/CortexTheseus/log"
	cli "gopkg.in/urfave/cli.v1"
	"os"
	"os/exec"
)

type Config struct {
}

//sudo npm install -g bittorrent-tracker
func main() {
	var conf Config
	app := cli.NewApp()
	app.Flags = []cli.Flag{}

	app.Action = func(c *cli.Context) error {
		run(&conf)
		return nil
	}
	err := app.Run(os.Args)
	if err != nil {
		log.Crit("fatal", "err", err)
	}
}

func run(conf *Config) int {
	cmd := exec.Command("bittorrent-tracker", "--port", "5008") //, "2&>1", "&")
	err := cmd.Run()
	if err != nil {
		log.Crit("cmd.Run() failed", "err", err)
	}
	fmt.Println("Cortex tracker started")
	return 1
}
