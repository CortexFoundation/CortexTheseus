package main

import (
	"github.com/CortexFoundation/CortexTheseus/log"
	"os/exec"
)

//sudo npm install -g bittorrent-tracker
func main() {
	cmd := exec.Command("bittorrent-tracker", "--port", "5008")
	out, err := cmd.CombinedOutput()
	if err != nil {
		log.Error("Tracker failed", "err", err)
	}
	log.Info("Tracker suc", "out", out)
}
