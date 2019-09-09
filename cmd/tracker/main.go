package main

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/params"
	cli "gopkg.in/urfave/cli.v1"
	"os"
	"os/exec"
	"sync"
	//        "sync/atomic"
)

type Config struct {
	wg sync.WaitGroup
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

	for _, port := range params.Tracker_ports {
		//cmd := exec.Command("bittorrent-tracker", "--port", port, "--stats", "false", "--http", "--silent") //, "2&>1", "&")
		log.Info("Tracker service starting", "port", port)
		conf.wg.Add(1)
		go func(p string) error {
			defer conf.wg.Done()
			log.Info("Tracker service starting", "port", p)
			//cmd := exec.Command("bittorrent-tracker", "--port", p, "--stats", "false", "--http", "--silent")
			cmd := exec.Command("bittorrent-tracker", "--port", p, "--http", "--silent")
			err := cmd.Run()

			return err
		}(port)
	}
	conf.wg.Wait()
	//log.crit("cmd.Start", "err", err)
	return nil
}
