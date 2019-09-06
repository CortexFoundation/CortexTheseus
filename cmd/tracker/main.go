package main

import (
	cli "gopkg.in/urfave/cli.v1"
	"os"
	"os/exec"
)

type Config struct {
}
// curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.34.0/install.sh | bash
// source ~/.bashrc
// nvm install node
// sudo npm install -g bittorrent-tracker
func main() {
	var conf Config
	app := cli.NewApp()
	app.Flags = []cli.Flag{}

	app.Action = func(c *cli.Context) error {
		err := run(&conf)
		return err
	}
	err := app.Run(os.Args)
	if err != nil {
		os.Exit(1)
	}
}

func run(conf *Config) error {
	cmd := exec.Command("bittorrent-tracker", "--port", "5008", "--stats", "false", "--http", "--silent") //, "2&>1", "&")
	err := cmd.Run()
	if err != nil {
		return err
	}
        //log.crit("cmd.Start", "err", err)
	return nil
}
