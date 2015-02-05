package p2p

// Protocol represents a P2P subprotocol implementation.
type Protocol struct {
	// Name should contain the official protocol name,
	// often a three-letter word.
	Name string

	// Version should contain the version number of the protocol.
	Version uint

	// Length should contain the number of message codes used
	// by the protocol.
	Length uint64

	// Run is called in a new groutine when the protocol has been
	// negotiated with a peer. It should read and write messages from
	// rw. The Payload for each message must be fully consumed.
	//
	// The peer connection is closed when Start returns. It should return
	// any protocol-level error (such as an I/O error) that is
	// encountered.
	Run func(peer *Peer, rw MsgReadWriter) error
}

func (p Protocol) cap() Cap {
	return Cap{p.Name, p.Version}
}

// Cap is the structure of a peer capability.
type Cap struct {
	Name    string
	Version uint
}

func (cap Cap) RlpData() interface{} {
	return []interface{}{cap.Name, cap.Version}
}

type capsByName []Cap

func (cs capsByName) Len() int           { return len(cs) }
func (cs capsByName) Less(i, j int) bool { return cs[i].Name < cs[j].Name }
func (cs capsByName) Swap(i, j int)      { cs[i], cs[j] = cs[j], cs[i] }
