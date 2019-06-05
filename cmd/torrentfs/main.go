package main

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/torrentfs"
	cli "gopkg.in/urfave/cli.v1"
	glog "log"
	"os"
	"os/signal"
	"syscall"
)

type Config struct {
	Host       string
	Port       int
	Dir        string
	TrackerURI string
	LogLevel   int
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
		cli.StringFlag{
			Name:        "host",
			Value:       "localhost",
			Usage:       "hostname",
			Destination: &conf.Host,
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
		cli.StringFlag{
			Name:        "tracker-uri",
			Value:       "http://torrent.cortexlabs.ai:5008/announce",
			Usage:       "tracker uri",
			Destination: &conf.TrackerURI,
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
		DataDir:         torrentfs.DefaultConfig.DataDir,
		Host:            torrentfs.DefaultConfig.Host,
		Port:            torrentfs.DefaultConfig.Port,
		DefaultTrackers: torrentfs.DefaultConfig.DefaultTrackers,
		SyncMode:        torrentfs.DefaultConfig.SyncMode,
		TestMode:        torrentfs.DefaultConfig.TestMode,
	}

	cfg.Host = conf.Host
	cfg.Port = conf.Port
	cfg.DataDir = conf.Dir

	tfs, _ := torrentfs.New(&cfg, "")
	tfs.Start(nil)
	c := make(chan os.Signal, 1)
	signal.Notify(c, syscall.SIGINT, syscall.SIGTERM)
	for {
		<-c
		tfs.Stop()
	}
	return 0
}
