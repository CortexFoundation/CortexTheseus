package mmsg

import (
	"errors"
	"net"
	"strings"

	"github.com/anacrolix/mmsg/socket"
)

// Considered MSG_DONTWAIT, but I think Go puts the socket into non-blocking
// mode in its runtime and it seems to do the right thing.
const flags = 0

type Conn struct {
	// If this is not nil, attempts to use batch APIs will be skipped automatically.
	err error
	s   *socket.Conn
	pr  PacketReader
}

type PacketReader interface {
	ReadFrom([]byte) (int, net.Addr, error)
}

// pr must implement net.Conn for mmsg to be enabled.
func NewConn(pr PacketReader) *Conn {
	ret := Conn{
		pr: pr,
	}
	nc, ok := pr.(net.Conn)
	if ok {
		ret.s, ret.err = socket.NewConn(nc)
	} else {
		ret.err = errors.New("mmsg.NewConn: not a net.Conn")
	}
	return &ret
}

func (me *Conn) recvMsgAsMsgs(ms []Message) (int, error) {
	err := me.RecvMsg(&ms[0])
	if err != nil {
		return 0, err
	}
	return 1, err
}

func (me *Conn) RecvMsgs(ms []Message) (n int, err error) {
	if me.err != nil || len(ms) == 1 {
		return me.recvMsgAsMsgs(ms)
	}
	sms := make([]socket.Message, len(ms))
	for i := range ms {
		sms[i].Buffers = ms[i].Buffers
	}
	n, err = me.s.RecvMsgs(sms, flags)
	if err != nil && strings.Contains(err.Error(), "not implemented") {
		if me.err != nil {
			panic(me.err)
		}
		me.err = err
		if n <= 0 {
			return me.recvMsgAsMsgs(ms)
		}
		err = nil
	}
	for i := 0; i < n; i++ {
		ms[i].Addr = sms[i].Addr
		ms[i].N = sms[i].N
	}
	return n, err
}

func (me *Conn) RecvMsg(m *Message) error {
	if len(m.Buffers) == 1 { // What about 0?
		var err error
		m.N, m.Addr, err = me.pr.ReadFrom(m.Buffers[0])
		return err
	}
	sm := socket.Message{
		Buffers: m.Buffers,
	}
	err := me.s.RecvMsg(&sm, flags)
	m.Addr = sm.Addr
	m.N = sm.N
	return err
}

type Message struct {
	Buffers [][]byte
	N       int
	Addr    net.Addr
}

func (me *Message) Payload() (p []byte) {
	n := me.N
	for _, b := range me.Buffers {
		if len(b) >= n {
			p = append(p, b[:n]...)
			return
		}
		p = append(p, b...)
		n -= len(b)
	}
	panic(n)
}

// Returns not nil if message batching is not working.
func (me *Conn) Err() error {
	return me.err
}
