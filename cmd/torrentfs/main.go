package main

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/torrentfs"
	cli "gopkg.in/urfave/cli.v1"
	glog "log"
	"os"
	"os/signal"
	"syscall"
)

type Config struct {
	Dir      string
	TaskList string
	LogLevel int
	NSeed    int
	NActive  int
	Dht      bool
}

var gitCommit = "" // Git SHA1 commit hash of the release (set via linker flags)

func main() {
	var conf Config
	app := cli.NewApp()

	app.Flags = []cli.Flag{
		cli.IntFlag{
			Name:        "verbosity",
			Value:       2,
			Usage:       "verbose level",
			Destination: &conf.LogLevel,
		},
		cli.IntFlag{
			Name:        "port",
			Value:       8085,
			Usage:       "port",
			Destination: &conf.LogLevel,
		},
		cli.StringFlag{
			Name:        "dir",
			Value:       "/data",
			Usage:       "datadir",
			Destination: &conf.Dir,
		},
	}

	app.Action = func(c *cli.Context) error {
		mainExitCode(&conf)
		return nil
	}

	err := app.Run(os.Args)
	if err != nil {
		glog.Fatal(err)
	}
}

func mainExitCode(conf *Config) int {
	log.Root().SetHandler(log.LvlFilterHandler(log.Lvl(conf.LogLevel), log.StreamHandler(os.Stdout, log.TerminalFormat(true))))

	cfg := torrentfs.Config{
		RpcURI:          "",
		DefaultTrackers: torrentfs.DefaultConfig.DefaultTrackers,
		BoostNodes:      torrentfs.DefaultConfig.BoostNodes,
		SyncMode:        torrentfs.DefaultConfig.SyncMode,
		DisableUTP:      torrentfs.DefaultConfig.DisableUTP,
		MaxSeedingNum:   conf.NSeed,
		MaxActiveNum:    conf.NActive,
	}

	cfg.DataDir = conf.Dir
	cfg.DisableDHT = !conf.Dht
	cfg.DisableUTP = true
	tfs, _ := torrentfs.New(&cfg, "", true, false)
	tfs.Start(nil)
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	for {
		<-c
		tfs.Stop()
	}
	return 0
}
