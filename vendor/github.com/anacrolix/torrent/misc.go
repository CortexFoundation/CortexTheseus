package torrent

import (
	"errors"
	"net"

	"github.com/RoaringBitmap/roaring"
	"github.com/anacrolix/missinggo/v2"
	"golang.org/x/time/rate"

	"github.com/anacrolix/torrent/metainfo"
	pp "github.com/anacrolix/torrent/peer_protocol"
	"github.com/anacrolix/torrent/types"
	"github.com/anacrolix/torrent/types/infohash"
)

type (
	// TODO: Make this private. Use types.Request in the (one?) place it's exposed here.
	Request       = types.Request
	ChunkSpec     = types.ChunkSpec
	PiecePriority = types.PiecePriority
)

const (
	PiecePriorityNormal    = types.PiecePriorityNormal
	PiecePriorityNone      = types.PiecePriorityNone
	PiecePriorityNow       = types.PiecePriorityNow
	PiecePriorityReadahead = types.PiecePriorityReadahead
	PiecePriorityNext      = types.PiecePriorityNext
	PiecePriorityHigh      = types.PiecePriorityHigh
)

func newRequest(index, begin, length pp.Integer) Request {
	return Request{index, ChunkSpec{begin, length}}
}

func newRequestFromMessage(msg *pp.Message) Request {
	switch msg.Type {
	case pp.Request, pp.Cancel, pp.Reject:
		return newRequest(msg.Index, msg.Begin, msg.Length)
	case pp.Piece:
		return newRequest(msg.Index, msg.Begin, pp.Integer(len(msg.Piece)))
	default:
		panic(msg.Type)
	}
}

// The size in bytes of a metadata extension piece.
func metadataPieceSize(totalSize, piece int) int {
	ret := totalSize - piece*(1<<14)
	if ret > 1<<14 {
		ret = 1 << 14
	}
	return ret
}

// Return the request that would include the given offset into the torrent data.
func torrentOffsetRequest(
	torrentLength, pieceSize, chunkSize, offset int64,
) (
	r Request, ok bool,
) {
	if offset < 0 || offset >= torrentLength {
		return
	}
	r.Index = pp.Integer(offset / pieceSize)
	r.Begin = pp.Integer(offset % pieceSize / chunkSize * chunkSize)
	r.Length = pp.Integer(chunkSize)
	pieceLeft := pp.Integer(pieceSize - int64(r.Begin))
	if r.Length > pieceLeft {
		r.Length = pieceLeft
	}
	torrentLeft := torrentLength - int64(r.Index)*pieceSize - int64(r.Begin)
	if int64(r.Length) > torrentLeft {
		r.Length = pp.Integer(torrentLeft)
	}
	ok = true
	return
}

func torrentRequestOffset(torrentLength, pieceSize int64, r Request) (off int64) {
	off = int64(r.Index)*pieceSize + int64(r.Begin)
	if off < 0 || off >= torrentLength {
		panic("invalid Request")
	}
	return
}

func validateInfo(info *metainfo.Info) error {
	if len(info.Pieces)%20 != 0 {
		return errors.New("pieces has invalid length")
	}
	if info.PieceLength == 0 {
		if info.TotalLength() != 0 {
			return errors.New("zero piece length")
		}
	} else if !info.HasV2() {
		// TotalLength returns different values for V1 and V2 depending on whether v1 pad files are
		// counted. Split the interface into several methods?
		if int((info.TotalLength()+info.PieceLength-1)/info.PieceLength) != info.NumPieces() {
			return errors.New("piece count and file lengths are at odds")
		}
	}
	return nil
}

func chunkIndexSpec(index, pieceLength, chunkSize pp.Integer) ChunkSpec {
	ret := ChunkSpec{index * chunkSize, chunkSize}
	if ret.Begin+ret.Length > pieceLength {
		ret.Length = pieceLength - ret.Begin
	}
	return ret
}

func comparePeerTrust(l, r *Peer) int {
	return l.trust().Cmp(r.trust())
}

func connIsIpv6(nc interface {
	LocalAddr() net.Addr
},
) bool {
	ra := nc.LocalAddr()
	rip := addrIpOrNil(ra)
	return rip.To4() == nil && rip.To16() != nil
}

var unlimited = rate.NewLimiter(rate.Inf, 0)

type (
	pieceIndex = int
	// Deprecated: Use infohash.T directly to avoid unnecessary imports.
	InfoHash = infohash.T
	IpPort   = missinggo.IpPort
)

func boolSliceToBitmap(slice []bool) (rb roaring.Bitmap) {
	for i, b := range slice {
		if b {
			rb.AddInt(i)
		}
	}
	return
}
