package torrent

import (
	"io"

	pp "github.com/anacrolix/torrent/peer_protocol"
)

// Various connection-level metrics. At the Torrent level these are aggregates. Chunks are messages
// with data payloads. Data is actual torrent content without any overhead. Useful is something we
// needed locally. Intended is something we were expecting (I think such as when we cancel a request
// but it arrives anyway). Written is things sent to the peer, and Read is stuff received from them.
// Due to the implementation of Count, must be aligned on some platforms: See
// https://github.com/anacrolix/torrent/issues/262.
type ConnStats struct {
	// Total bytes on the wire. Includes handshakes and encryption.
	BytesWritten     Count
	BytesWrittenData Count

	BytesRead                   Count
	BytesReadData               Count
	BytesReadUsefulData         Count
	BytesReadUsefulIntendedData Count

	ChunksWritten Count

	ChunksRead       Count
	ChunksReadUseful Count
	ChunksReadWasted Count

	MetadataChunksRead Count

	// Number of pieces data was written to, that subsequently passed verification.
	PiecesDirtiedGood Count
	// Number of pieces data was written to, that subsequently failed verification. Note that a
	// connection may not have been the sole dirtier of a piece.
	PiecesDirtiedBad Count
}

func (me *ConnStats) Copy() (ret ConnStats) {
	return copyCountFields(me)
}

func (cs *ConnStats) wroteMsg(msg *pp.Message) {
	// TODO: Track messages and not just chunks.
	switch msg.Type {
	case pp.Piece:
		cs.ChunksWritten.Add(1)
		cs.BytesWrittenData.Add(int64(len(msg.Piece)))
	}
}

func (cs *ConnStats) receivedChunk(size int64) {
	cs.ChunksRead.Add(1)
	cs.BytesReadData.Add(size)
}

func (cs *ConnStats) incrementPiecesDirtiedGood() {
	cs.PiecesDirtiedGood.Add(1)
}

func (cs *ConnStats) incrementPiecesDirtiedBad() {
	cs.PiecesDirtiedBad.Add(1)
}

func add(n int64, f func(*ConnStats) *Count) func(*ConnStats) {
	return func(cs *ConnStats) {
		p := f(cs)
		p.Add(n)
	}
}

type connStatsReadWriter struct {
	rw io.ReadWriter
	c  *PeerConn
}

func (me connStatsReadWriter) Write(b []byte) (n int, err error) {
	n, err = me.rw.Write(b)
	me.c.wroteBytes(int64(n))
	return
}

func (me connStatsReadWriter) Read(b []byte) (n int, err error) {
	n, err = me.rw.Read(b)
	me.c.readBytes(int64(n))
	return
}
