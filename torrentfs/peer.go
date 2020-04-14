package torrentfs

import (
	//"github.com/CortexFoundation/CortexTheseus/log"
	//"github.com/CortexFoundation/CortexTheseus/params"
	//"github.com/CortexFoundation/CortexTheseus/rpc"
	//"github.com/anacrolix/torrent/metainfo"
	mapset "github.com/deckarep/golang-set"
	//"io/ioutil"
	//"path"
	//"sync"
	//"time"
	//"strings"
	//"errors"
	//"github.com/CortexFoundation/CortexTheseus/common/compress"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	//"github.com/CortexFoundation/CortexTheseus/p2p/enode"
)

type Peer struct {
	host *TorrentFS
	peer *p2p.Peer
	ws   p2p.MsgReadWriter

	trusted bool

	known mapset.Set // Messages already known by the peer to avoid wasting bandwidth
}

func newPeer(host *TorrentFS, remote *p2p.Peer, rw p2p.MsgReadWriter) *Peer {
	return &Peer{
		host:    host,
		peer:    remote,
		ws:      rw,
		trusted: false,
		known:   mapset.NewSet(),
	}
}
