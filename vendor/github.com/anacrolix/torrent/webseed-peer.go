package torrent

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/RoaringBitmap/roaring"
	"github.com/anacrolix/log"
	"github.com/anacrolix/torrent/metainfo"
	pp "github.com/anacrolix/torrent/peer_protocol"
	"github.com/anacrolix/torrent/webseed"
)

type webseedPeer struct {
	client         webseed.Client
	activeRequests map[Request]webseed.Request
	requesterCond  sync.Cond
	peer           Peer
	// Number of requester routines.
	maxRequests int
}

var _ peerImpl = (*webseedPeer)(nil)

func (me *webseedPeer) connStatusString() string {
	return me.client.Url
}

func (ws *webseedPeer) String() string {
	return fmt.Sprintf("webseed peer for %q", ws.client.Url)
}

func (ws *webseedPeer) onGotInfo(info *metainfo.Info) {
	ws.client.SetInfo(info)
	// There should be probably be a callback in Client instead, so it can remove pieces at its whim
	// too.
	ws.client.Pieces.Iterate(func(x uint32) bool {
		ws.peer.t.incPieceAvailability(pieceIndex(x))
		return true
	})
}

func (ws *webseedPeer) writeInterested(interested bool) bool {
	return true
}

func (ws *webseedPeer) _cancel(r RequestIndex) bool {
	active, ok := ws.activeRequests[ws.peer.t.requestIndexToRequest(r)]
	if ok {
		active.Cancel()
		if !ws.peer.deleteRequest(r) {
			panic("cancelled webseed request should exist")
		}
		if ws.peer.isLowOnRequests() {
			ws.peer.updateRequests("webseedPeer._cancel")
		}
	}
	return true
}

func (ws *webseedPeer) intoSpec(r Request) webseed.RequestSpec {
	return webseed.RequestSpec{ws.peer.t.requestOffset(r), int64(r.Length)}
}

func (ws *webseedPeer) _request(r Request) bool {
	ws.requesterCond.Signal()
	return true
}

func (ws *webseedPeer) doRequest(r Request) {
	webseedRequest := ws.client.NewRequest(ws.intoSpec(r))
	ws.activeRequests[r] = webseedRequest
	func() {
		ws.requesterCond.L.Unlock()
		defer ws.requesterCond.L.Lock()
		ws.requestResultHandler(r, webseedRequest)
	}()
	delete(ws.activeRequests, r)
}

func (ws *webseedPeer) requester() {
	ws.requesterCond.L.Lock()
	defer ws.requesterCond.L.Unlock()
start:
	for !ws.peer.closed.IsSet() {
		restart := false
		ws.peer.actualRequestState.Requests.Iterate(func(x uint32) bool {
			r := ws.peer.t.requestIndexToRequest(x)
			if _, ok := ws.activeRequests[r]; ok {
				return true
			}
			ws.doRequest(r)
			restart = true
			return false
		})
		if restart {
			goto start
		}
		ws.requesterCond.Wait()
	}
}

func (ws *webseedPeer) connectionFlags() string {
	return "WS"
}

// TODO: This is called when banning peers. Perhaps we want to be able to ban webseeds too. We could
// return bool if this is even possible, and if it isn't, skip to the next drop candidate.
func (ws *webseedPeer) drop() {}

func (ws *webseedPeer) handleUpdateRequests() {
	ws.peer.maybeUpdateActualRequestState()
}

func (ws *webseedPeer) onClose() {
	ws.peer.logger.WithLevel(log.Debug).Print("closing")
	ws.peer.deleteAllRequests()
	for _, r := range ws.activeRequests {
		r.Cancel()
	}
	ws.requesterCond.Broadcast()
}

func (ws *webseedPeer) requestResultHandler(r Request, webseedRequest webseed.Request) {
	result := <-webseedRequest.Result
	// We do this here rather than inside receiveChunk, since we want to count errors too. I'm not
	// sure if we can divine which errors indicate cancellation on our end without hitting the
	// network though.
	ws.peer.doChunkReadStats(int64(len(result.Bytes)))
	ws.peer.readBytes(int64(len(result.Bytes)))
	ws.peer.t.cl.lock()
	defer ws.peer.t.cl.unlock()
	if result.Err != nil {
		if !errors.Is(result.Err, context.Canceled) && !ws.peer.closed.IsSet() {
			ws.peer.logger.Printf("Request %v rejected: %v", r, result.Err)
			// cfg := spew.NewDefaultConfig()
			// cfg.DisableMethods = true
			// cfg.Dump(result.Err)
			log.Printf("closing %v", ws)
			ws.peer.close()
		}
		ws.peer.remoteRejectedRequest(ws.peer.t.requestIndexFromRequest(r))
	} else {
		err := ws.peer.receiveChunk(&pp.Message{
			Type:  pp.Piece,
			Index: r.Index,
			Begin: r.Begin,
			Piece: result.Bytes,
		})
		if err != nil {
			panic(err)
		}
	}
}

func (me *webseedPeer) isLowOnRequests() bool {
	return me.peer.actualRequestState.Requests.GetCardinality() < uint64(me.maxRequests)
}

func (me *webseedPeer) peerPieces() *roaring.Bitmap {
	return &me.client.Pieces
}

func (cn *webseedPeer) peerHasAllPieces() (all, known bool) {
	if !cn.peer.t.haveInfo() {
		return true, false
	}
	return cn.client.Pieces.GetCardinality() == uint64(cn.peer.t.numPieces()), true
}
