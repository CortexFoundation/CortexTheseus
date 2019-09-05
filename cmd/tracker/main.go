package main

import (
	//"fmt"
	"github.com/CortexFoundation/CortexTheseus/log"
	cli "gopkg.in/urfave/cli.v1"
	"os"
	"os/exec"
	"syscall"
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
	if err := cmd.Start(); err != nil {
        //log.crit("cmd.Start", "err", err)
    }

    if err := cmd.Wait(); err != nil {
        if exiterr, ok := err.(*exec.ExitError); ok {
            // The program has exited with an exit code != 0

            // This works on both Unix and Windows. Although package
            // syscall is generally platform dependent, WaitStatus is
            // defined for both Unix and Windows and in both cases has
            // an ExitStatus() method with the same signature.
            if status, ok := exiterr.Sys().(syscall.WaitStatus); ok {
                log.Info("Exit Status","status", status.ExitStatus())
            }
        } else {
            log.Crit("cmd.Wait", "err", err)
        }
    }
	return 1
}
