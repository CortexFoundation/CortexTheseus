// Copyright 2020 The CortexTheseus Authors
// This file is part of the CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

package params

// Config ...
type Config struct {
	// Host is the host interface on which to start the storage server. If this
	// field is empty, no storage will be started.
	Port            int      `toml:",omitempty"`
	DataDir         string   `toml:",omitempty"`
	RpcURI          string   `toml:",omitempty"`
	IpcPath         string   `toml:",omitempty"`
	DisableUTP      bool     `toml:",omitempty"`
	DisableTCP      bool     `toml:",omitempty"`
	DisableDHT      bool     `toml:",omitempty"`
	DisableIPv6     bool     `toml:",omitempty"`
	DefaultTrackers []string `toml:",omitempty"`
	BoostNodes      []string `toml:",omitempty"`
	Mode            string   `toml:",omitempty"`
	MaxSeedingNum   int      `toml:",omitempty"`
	MaxActiveNum    int      `toml:",omitempty"`
	//FullSeed        bool     `toml:",omitempty"`
	Boost        bool   `toml:",omitempty"`
	Quiet        bool   `toml:",omitempty"`
	UploadRate   int    `toml:",omitempty"`
	DownloadRate int    `toml:",omitempty"`
	Metrics      bool   `toml:",omitempty"`
	Server       bool   `toml:",omitempty"`
	Wormhole     bool   `toml:",omitempty"`
	Engine       string `toml:",omitempty"`
}

// DefaultConfig contains default settings for the storage.
var DefaultConfig = Config{
	Port:            0,
	DefaultTrackers: MainnetTrackers,
	BoostNodes:      TorrentBoostNodes,
	Mode:            "default",
	DisableUTP:      true,
	DisableDHT:      false,
	DisableTCP:      false,
	DisableIPv6:     false,
	MaxSeedingNum:   LimitSeeding / 2,
	MaxActiveNum:    LimitSeeding / 2,
	//FullSeed:        false,
	Boost:        false,
	Quiet:        true,
	UploadRate:   -1,
	DownloadRate: -1,
	Metrics:      true,
	Server:       false,
	Wormhole:     false,
	Engine:       "badger",
}

const (
	QueryTimeInterval              = 1
	ExpansionFactor        float64 = 1.2
	DefaultSeedInterval            = 600
	TorrentWaitingTime             = 1800
	DownloadWaitingTime            = 2700
	DefaultBytesLimitation         = 512 * 1024
	DefaultTmpPath                 = ".tmp"
	Version                        = "3"
)
