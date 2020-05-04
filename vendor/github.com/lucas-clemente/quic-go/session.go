package quic

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"reflect"
	"sync"
	"time"

	"github.com/lucas-clemente/quic-go/internal/ackhandler"
	"github.com/lucas-clemente/quic-go/internal/congestion"
	"github.com/lucas-clemente/quic-go/internal/flowcontrol"
	"github.com/lucas-clemente/quic-go/internal/handshake"
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/internal/wire"
	"github.com/lucas-clemente/quic-go/qlog"
	"github.com/lucas-clemente/quic-go/quictrace"
)

type unpacker interface {
	Unpack(hdr *wire.Header, rcvTime time.Time, data []byte) (*unpackedPacket, error)
}

type streamGetter interface {
	GetOrOpenReceiveStream(protocol.StreamID) (receiveStreamI, error)
	GetOrOpenSendStream(protocol.StreamID) (sendStreamI, error)
}

type streamManager interface {
	GetOrOpenSendStream(protocol.StreamID) (sendStreamI, error)
	GetOrOpenReceiveStream(protocol.StreamID) (receiveStreamI, error)
	OpenStream() (Stream, error)
	OpenUniStream() (SendStream, error)
	OpenStreamSync(context.Context) (Stream, error)
	OpenUniStreamSync(context.Context) (SendStream, error)
	AcceptStream(context.Context) (Stream, error)
	AcceptUniStream(context.Context) (ReceiveStream, error)
	DeleteStream(protocol.StreamID) error
	UpdateLimits(*wire.TransportParameters) error
	HandleMaxStreamsFrame(*wire.MaxStreamsFrame) error
	CloseWithError(error)
}

type cryptoStreamHandler interface {
	RunHandshake()
	ChangeConnectionID(protocol.ConnectionID)
	SetLargest1RTTAcked(protocol.PacketNumber)
	DropHandshakeKeys()
	GetSessionTicket() ([]byte, error)
	io.Closer
	ConnectionState() handshake.ConnectionState
}

type receivedPacket struct {
	remoteAddr net.Addr
	rcvTime    time.Time
	data       []byte

	buffer *packetBuffer
}

func (p *receivedPacket) Clone() *receivedPacket {
	return &receivedPacket{
		remoteAddr: p.remoteAddr,
		rcvTime:    p.rcvTime,
		data:       p.data,
		buffer:     p.buffer,
	}
}

type sessionRunner interface {
	Add(protocol.ConnectionID, packetHandler) bool
	GetStatelessResetToken(protocol.ConnectionID) [16]byte
	Retire(protocol.ConnectionID)
	Remove(protocol.ConnectionID)
	ReplaceWithClosed(protocol.ConnectionID, packetHandler)
	AddResetToken([16]byte, packetHandler)
	RemoveResetToken([16]byte)
	RetireResetToken([16]byte)
}

type handshakeRunner struct {
	onReceivedParams    func(*wire.TransportParameters)
	onError             func(error)
	dropKeys            func(protocol.EncryptionLevel)
	onHandshakeComplete func()
}

func (r *handshakeRunner) OnReceivedParams(tp *wire.TransportParameters) { r.onReceivedParams(tp) }
func (r *handshakeRunner) OnError(e error)                               { r.onError(e) }
func (r *handshakeRunner) DropKeys(el protocol.EncryptionLevel)          { r.dropKeys(el) }
func (r *handshakeRunner) OnHandshakeComplete()                          { r.onHandshakeComplete() }

type closeError struct {
	err       error
	remote    bool
	immediate bool
}

var errCloseForRecreating = errors.New("closing session in order to recreate it")

// A Session is a QUIC session
type session struct {
	// Destination connection ID used during the handshake.
	// Used to check source connection ID on incoming packets.
	handshakeDestConnID protocol.ConnectionID
	// if the server sends a Retry, this is the connection ID we used initially
	origDestConnID protocol.ConnectionID
	srcConnIDLen   int

	perspective    protocol.Perspective
	initialVersion protocol.VersionNumber // if version negotiation is performed, this is the version we initially tried
	version        protocol.VersionNumber
	config         *Config

	conn      connection
	sendQueue *sendQueue

	streamsMap      streamManager
	connIDManager   *connIDManager
	connIDGenerator *connIDGenerator

	rttStats *congestion.RTTStats

	cryptoStreamManager   *cryptoStreamManager
	sentPacketHandler     ackhandler.SentPacketHandler
	receivedPacketHandler ackhandler.ReceivedPacketHandler
	retransmissionQueue   *retransmissionQueue
	framer                framer
	windowUpdateQueue     *windowUpdateQueue
	connFlowController    flowcontrol.ConnectionFlowController
	tokenStoreKey         string                    // only set for the client
	tokenGenerator        *handshake.TokenGenerator // only set for the server

	unpacker    unpacker
	frameParser wire.FrameParser
	packer      packer

	oneRTTStream        cryptoStream // only set for the server
	cryptoStreamHandler cryptoStreamHandler

	receivedPackets  chan *receivedPacket
	sendingScheduled chan struct{}

	closeOnce sync.Once
	// closeChan is used to notify the run loop that it should terminate
	closeChan chan closeError

	ctx                context.Context
	ctxCancel          context.CancelFunc
	handshakeCtx       context.Context
	handshakeCtxCancel context.CancelFunc

	undecryptablePackets []*receivedPacket

	clientHelloWritten    <-chan *wire.TransportParameters
	earlySessionReadyChan chan struct{}
	handshakeCompleteChan chan struct{} // is closed when the handshake completes
	handshakeComplete     bool
	handshakeConfirmed    bool

	receivedRetry       bool
	receivedFirstPacket bool

	idleTimeout         time.Duration
	sessionCreationTime time.Time
	// The idle timeout is set based on the max of the time we received the last packet...
	lastPacketReceivedTime time.Time
	// ... and the time we sent a new ack-eliciting packet after receiving a packet.
	firstAckElicitingPacketAfterIdleSentTime time.Time
	// pacingDeadline is the time when the next packet should be sent
	pacingDeadline time.Time

	peerParams *wire.TransportParameters

	timer *utils.Timer
	// keepAlivePingSent stores whether a keep alive PING is in flight.
	// It is reset as soon as we receive a packet from the peer.
	keepAlivePingSent bool
	keepAliveInterval time.Duration

	traceCallback func(quictrace.Event)

	logID   string
	qlogger qlog.Tracer
	logger  utils.Logger
}

var _ Session = &session{}
var _ EarlySession = &session{}
var _ streamSender = &session{}

var newSession = func(
	conn connection,
	runner sessionRunner,
	origDestConnID protocol.ConnectionID,
	clientDestConnID protocol.ConnectionID,
	destConnID protocol.ConnectionID,
	srcConnID protocol.ConnectionID,
	statelessResetToken [16]byte,
	conf *Config,
	tlsConf *tls.Config,
	tokenGenerator *handshake.TokenGenerator,
	enable0RTT bool,
	qlogger qlog.Tracer,
	logger utils.Logger,
	v protocol.VersionNumber,
) quicSession {
	s := &session{
		conn:                  conn,
		config:                conf,
		handshakeDestConnID:   destConnID,
		srcConnIDLen:          srcConnID.Len(),
		tokenGenerator:        tokenGenerator,
		oneRTTStream:          newCryptoStream(),
		perspective:           protocol.PerspectiveServer,
		handshakeCompleteChan: make(chan struct{}),
		qlogger:               qlogger,
		logger:                logger,
		version:               v,
	}
	if origDestConnID != nil {
		s.logID = origDestConnID.String()
	} else {
		s.logID = destConnID.String()
	}
	s.connIDManager = newConnIDManager(
		destConnID,
		func(token [16]byte) { runner.AddResetToken(token, s) },
		runner.RemoveResetToken,
		runner.RetireResetToken,
		s.queueControlFrame,
	)
	s.connIDGenerator = newConnIDGenerator(
		srcConnID,
		clientDestConnID,
		func(connID protocol.ConnectionID) { runner.Add(connID, s) },
		runner.GetStatelessResetToken,
		runner.Remove,
		runner.Retire,
		runner.ReplaceWithClosed,
		s.queueControlFrame,
	)
	s.preSetup()
	s.sentPacketHandler, s.receivedPacketHandler = ackhandler.NewAckHandler(
		0,
		s.rttStats,
		s.perspective,
		s.traceCallback,
		s.qlogger,
		s.logger,
		s.version,
	)
	initialStream := newCryptoStream()
	handshakeStream := newCryptoStream()
	params := &wire.TransportParameters{
		InitialMaxStreamDataBidiLocal:  protocol.InitialMaxStreamData,
		InitialMaxStreamDataBidiRemote: protocol.InitialMaxStreamData,
		InitialMaxStreamDataUni:        protocol.InitialMaxStreamData,
		InitialMaxData:                 protocol.InitialMaxData,
		MaxIdleTimeout:                 s.config.MaxIdleTimeout,
		MaxBidiStreamNum:               protocol.StreamNum(s.config.MaxIncomingStreams),
		MaxUniStreamNum:                protocol.StreamNum(s.config.MaxIncomingUniStreams),
		MaxAckDelay:                    protocol.MaxAckDelayInclGranularity,
		AckDelayExponent:               protocol.AckDelayExponent,
		DisableActiveMigration:         true,
		StatelessResetToken:            &statelessResetToken,
		OriginalConnectionID:           origDestConnID,
		ActiveConnectionIDLimit:        protocol.MaxActiveConnectionIDs,
	}
	if s.qlogger != nil {
		s.qlogger.SentTransportParameters(params)
	}
	cs := handshake.NewCryptoSetupServer(
		initialStream,
		handshakeStream,
		clientDestConnID,
		conn.LocalAddr(),
		conn.RemoteAddr(),
		params,
		&handshakeRunner{
			onReceivedParams: s.processTransportParameters,
			onError:          s.closeLocal,
			dropKeys:         s.dropEncryptionLevel,
			onHandshakeComplete: func() {
				runner.Retire(clientDestConnID)
				close(s.handshakeCompleteChan)
			},
		},
		tlsConf,
		enable0RTT,
		s.rttStats,
		qlogger,
		logger,
	)
	s.cryptoStreamHandler = cs
	s.packer = newPacketPacker(
		srcConnID,
		s.connIDManager.Get,
		initialStream,
		handshakeStream,
		s.sentPacketHandler,
		s.retransmissionQueue,
		s.RemoteAddr(),
		cs,
		s.framer,
		s.receivedPacketHandler,
		s.perspective,
		s.version,
	)
	s.unpacker = newPacketUnpacker(cs, s.version)
	s.cryptoStreamManager = newCryptoStreamManager(cs, initialStream, handshakeStream, s.oneRTTStream)
	return s
}

// declare this as a variable, such that we can it mock it in the tests
var newClientSession = func(
	conn connection,
	runner sessionRunner,
	destConnID protocol.ConnectionID,
	srcConnID protocol.ConnectionID,
	conf *Config,
	tlsConf *tls.Config,
	initialPacketNumber protocol.PacketNumber,
	initialVersion protocol.VersionNumber,
	enable0RTT bool,
	qlogger qlog.Tracer,
	logger utils.Logger,
	v protocol.VersionNumber,
) quicSession {
	s := &session{
		conn:                  conn,
		config:                conf,
		handshakeDestConnID:   destConnID,
		srcConnIDLen:          srcConnID.Len(),
		perspective:           protocol.PerspectiveClient,
		handshakeCompleteChan: make(chan struct{}),
		logID:                 destConnID.String(),
		logger:                logger,
		qlogger:               qlogger,
		initialVersion:        initialVersion,
		version:               v,
	}
	s.connIDManager = newConnIDManager(
		destConnID,
		func(token [16]byte) { runner.AddResetToken(token, s) },
		runner.RemoveResetToken,
		runner.RetireResetToken,
		s.queueControlFrame,
	)
	s.connIDGenerator = newConnIDGenerator(
		srcConnID,
		nil,
		func(connID protocol.ConnectionID) { runner.Add(connID, s) },
		runner.GetStatelessResetToken,
		runner.Remove,
		runner.Retire,
		runner.ReplaceWithClosed,
		s.queueControlFrame,
	)
	s.preSetup()
	s.sentPacketHandler, s.receivedPacketHandler = ackhandler.NewAckHandler(
		initialPacketNumber,
		s.rttStats,
		s.perspective,
		s.traceCallback,
		s.qlogger,
		s.logger,
		s.version,
	)
	initialStream := newCryptoStream()
	handshakeStream := newCryptoStream()
	params := &wire.TransportParameters{
		InitialMaxStreamDataBidiRemote: protocol.InitialMaxStreamData,
		InitialMaxStreamDataBidiLocal:  protocol.InitialMaxStreamData,
		InitialMaxStreamDataUni:        protocol.InitialMaxStreamData,
		InitialMaxData:                 protocol.InitialMaxData,
		MaxIdleTimeout:                 s.config.MaxIdleTimeout,
		MaxBidiStreamNum:               protocol.StreamNum(s.config.MaxIncomingStreams),
		MaxUniStreamNum:                protocol.StreamNum(s.config.MaxIncomingUniStreams),
		MaxAckDelay:                    protocol.MaxAckDelayInclGranularity,
		AckDelayExponent:               protocol.AckDelayExponent,
		DisableActiveMigration:         true,
		ActiveConnectionIDLimit:        protocol.MaxActiveConnectionIDs,
	}
	if s.qlogger != nil {
		s.qlogger.SentTransportParameters(params)
	}
	cs, clientHelloWritten := handshake.NewCryptoSetupClient(
		initialStream,
		handshakeStream,
		destConnID,
		conn.LocalAddr(),
		conn.RemoteAddr(),
		params,
		&handshakeRunner{
			onReceivedParams:    s.processTransportParameters,
			onError:             s.closeLocal,
			dropKeys:            s.dropEncryptionLevel,
			onHandshakeComplete: func() { close(s.handshakeCompleteChan) },
		},
		tlsConf,
		enable0RTT,
		s.rttStats,
		qlogger,
		logger,
	)
	s.clientHelloWritten = clientHelloWritten
	s.cryptoStreamHandler = cs
	s.cryptoStreamManager = newCryptoStreamManager(cs, initialStream, handshakeStream, newCryptoStream())
	s.unpacker = newPacketUnpacker(cs, s.version)
	s.packer = newPacketPacker(
		srcConnID,
		s.connIDManager.Get,
		initialStream,
		handshakeStream,
		s.sentPacketHandler,
		s.retransmissionQueue,
		s.RemoteAddr(),
		cs,
		s.framer,
		s.receivedPacketHandler,
		s.perspective,
		s.version,
	)
	if len(tlsConf.ServerName) > 0 {
		s.tokenStoreKey = tlsConf.ServerName
	} else {
		s.tokenStoreKey = conn.RemoteAddr().String()
	}
	if s.config.TokenStore != nil {
		if token := s.config.TokenStore.Pop(s.tokenStoreKey); token != nil {
			s.packer.SetToken(token.data)
		}
	}
	return s
}

func (s *session) preSetup() {
	s.sendQueue = newSendQueue(s.conn)
	s.retransmissionQueue = newRetransmissionQueue(s.version)
	s.frameParser = wire.NewFrameParser(s.version)
	s.rttStats = &congestion.RTTStats{}
	s.connFlowController = flowcontrol.NewConnectionFlowController(
		protocol.InitialMaxData,
		protocol.ByteCount(s.config.MaxReceiveConnectionFlowControlWindow),
		s.onHasConnectionWindowUpdate,
		s.rttStats,
		s.logger,
	)
	s.earlySessionReadyChan = make(chan struct{})
	s.streamsMap = newStreamsMap(
		s,
		s.newFlowController,
		uint64(s.config.MaxIncomingStreams),
		uint64(s.config.MaxIncomingUniStreams),
		s.perspective,
		s.version,
	)
	s.framer = newFramer(s.streamsMap, s.version)
	s.receivedPackets = make(chan *receivedPacket, protocol.MaxSessionUnprocessedPackets)
	s.closeChan = make(chan closeError, 1)
	s.sendingScheduled = make(chan struct{}, 1)
	s.undecryptablePackets = make([]*receivedPacket, 0, protocol.MaxUndecryptablePackets)
	s.ctx, s.ctxCancel = context.WithCancel(context.Background())
	s.handshakeCtx, s.handshakeCtxCancel = context.WithCancel(context.Background())

	now := time.Now()
	s.lastPacketReceivedTime = now
	s.sessionCreationTime = now

	s.windowUpdateQueue = newWindowUpdateQueue(s.streamsMap, s.connFlowController, s.framer.QueueControlFrame)

	if s.config.QuicTracer != nil {
		s.traceCallback = func(ev quictrace.Event) {
			s.config.QuicTracer.Trace(s.origDestConnID, ev)
		}
	}
}

// run the session main loop
func (s *session) run() error {
	defer s.ctxCancel()

	s.timer = utils.NewTimer()

	go s.cryptoStreamHandler.RunHandshake()
	go func() {
		if err := s.sendQueue.Run(); err != nil {
			s.closeLocal(err)
		}
	}()

	if s.perspective == protocol.PerspectiveClient {
		select {
		case zeroRTTParams := <-s.clientHelloWritten:
			s.scheduleSending()
			if zeroRTTParams != nil {
				s.processTransportParameters(zeroRTTParams)
				close(s.earlySessionReadyChan)
			}
		case closeErr := <-s.closeChan:
			// put the close error back into the channel, so that the run loop can receive it
			s.closeChan <- closeErr
		}
	}

	var closeErr closeError

runLoop:
	for {
		// Close immediately if requested
		select {
		case closeErr = <-s.closeChan:
			break runLoop
		case <-s.handshakeCompleteChan:
			s.handleHandshakeComplete()
		default:
		}

		s.maybeResetTimer()

		select {
		case closeErr = <-s.closeChan:
			break runLoop
		case <-s.timer.Chan():
			s.timer.SetRead()
			// We do all the interesting stuff after the switch statement, so
			// nothing to see here.
		case <-s.sendingScheduled:
			// We do all the interesting stuff after the switch statement, so
			// nothing to see here.
		case p := <-s.receivedPackets:
			// Only reset the timers if this packet was actually processed.
			// This avoids modifying any state when handling undecryptable packets,
			// which could be injected by an attacker.
			if wasProcessed := s.handlePacketImpl(p); !wasProcessed {
				continue
			}
			// Don't set timers and send packets if the packet made us close the session.
			select {
			case closeErr = <-s.closeChan:
				break runLoop
			default:
			}
		case <-s.handshakeCompleteChan:
			s.handleHandshakeComplete()
		}

		now := time.Now()
		if timeout := s.sentPacketHandler.GetLossDetectionTimeout(); !timeout.IsZero() && timeout.Before(now) {
			// This could cause packets to be retransmitted.
			// Check it before trying to send packets.
			if err := s.sentPacketHandler.OnLossDetectionTimeout(); err != nil {
				s.closeLocal(err)
			}
		}

		var pacingDeadline time.Time
		if s.pacingDeadline.IsZero() { // the timer didn't have a pacing deadline set
			pacingDeadline = s.sentPacketHandler.TimeUntilSend()
		}
		if keepAliveTime := s.nextKeepAliveTime(); !keepAliveTime.IsZero() && !now.Before(keepAliveTime) {
			// send a PING frame since there is no activity in the session
			s.logger.Debugf("Sending a keep-alive PING to keep the connection alive.")
			s.framer.QueueControlFrame(&wire.PingFrame{})
			s.keepAlivePingSent = true
		} else if !s.handshakeComplete && now.Sub(s.sessionCreationTime) >= s.config.HandshakeTimeout {
			if s.qlogger != nil {
				s.qlogger.ClosedConnection(qlog.CloseReasonHandshakeTimeout)
			}
			s.destroyImpl(qerr.NewTimeoutError("Handshake did not complete in time"))
			continue
		} else if s.handshakeComplete && now.Sub(s.idleTimeoutStartTime()) >= s.idleTimeout {
			if s.qlogger != nil {
				s.qlogger.ClosedConnection(qlog.CloseReasonIdleTimeout)
			}
			s.destroyImpl(qerr.NewTimeoutError("No recent network activity"))
			continue
		} else if !pacingDeadline.IsZero() && now.Before(pacingDeadline) {
			// If we get to this point before the pacing deadline, we should wait until that deadline.
			// This can happen when scheduleSending is called, or a packet is received.
			// Set the timer and restart the run loop.
			s.pacingDeadline = pacingDeadline
			continue
		}

		if err := s.sendPackets(); err != nil {
			s.closeLocal(err)
		}
	}

	s.handleCloseError(closeErr)
	if closeErr.err != errCloseForRecreating && s.qlogger != nil {
		if err := s.qlogger.Export(); err != nil {
			s.logger.Errorf("exporting qlog failed: %s", err)
		}
	}
	s.logger.Infof("Connection %s closed.", s.logID)
	s.cryptoStreamHandler.Close()
	s.sendQueue.Close()
	s.timer.Stop()
	return closeErr.err
}

// blocks until the early session can be used
func (s *session) earlySessionReady() <-chan struct{} {
	return s.earlySessionReadyChan
}

func (s *session) HandshakeComplete() context.Context {
	return s.handshakeCtx
}

func (s *session) Context() context.Context {
	return s.ctx
}

func (s *session) ConnectionState() ConnectionState {
	return s.cryptoStreamHandler.ConnectionState()
}

// Time when the next keep-alive packet should be sent.
// It returns a zero time if no keep-alive should be sent.
func (s *session) nextKeepAliveTime() time.Time {
	if !s.config.KeepAlive || s.keepAlivePingSent || !s.firstAckElicitingPacketAfterIdleSentTime.IsZero() {
		return time.Time{}
	}
	return s.lastPacketReceivedTime.Add(s.keepAliveInterval / 2)
}

func (s *session) maybeResetTimer() {
	var deadline time.Time
	if !s.handshakeComplete {
		deadline = s.sessionCreationTime.Add(s.config.HandshakeTimeout)
	} else {
		if keepAliveTime := s.nextKeepAliveTime(); !keepAliveTime.IsZero() {
			deadline = keepAliveTime
		} else {
			deadline = s.idleTimeoutStartTime().Add(s.idleTimeout)
		}
	}

	if ackAlarm := s.receivedPacketHandler.GetAlarmTimeout(); !ackAlarm.IsZero() {
		deadline = utils.MinTime(deadline, ackAlarm)
	}
	if lossTime := s.sentPacketHandler.GetLossDetectionTimeout(); !lossTime.IsZero() {
		deadline = utils.MinTime(deadline, lossTime)
	}
	if !s.pacingDeadline.IsZero() {
		deadline = utils.MinTime(deadline, s.pacingDeadline)
	}

	s.timer.Reset(deadline)
}

func (s *session) idleTimeoutStartTime() time.Time {
	return utils.MaxTime(s.lastPacketReceivedTime, s.firstAckElicitingPacketAfterIdleSentTime)
}

func (s *session) handleHandshakeComplete() {
	s.handshakeComplete = true
	s.handshakeCompleteChan = nil // prevent this case from ever being selected again
	s.handshakeCtxCancel()

	s.connIDGenerator.SetHandshakeComplete()
	s.sentPacketHandler.SetHandshakeComplete()

	if s.perspective == protocol.PerspectiveServer {
		ticket, err := s.cryptoStreamHandler.GetSessionTicket()
		if err != nil {
			s.closeLocal(err)
		}
		if ticket != nil {
			s.oneRTTStream.Write(ticket)
			for s.oneRTTStream.HasData() {
				s.queueControlFrame(s.oneRTTStream.PopCryptoFrame(protocol.MaxPostHandshakeCryptoFrameSize))
			}
		}
		token, err := s.tokenGenerator.NewToken(s.conn.RemoteAddr())
		if err != nil {
			s.closeLocal(err)
		}
		s.queueControlFrame(&wire.NewTokenFrame{Token: token})
		s.cryptoStreamHandler.DropHandshakeKeys()
		s.queueControlFrame(&wire.HandshakeDoneFrame{})
	}
}

func (s *session) handlePacketImpl(rp *receivedPacket) bool {
	var counter uint8
	var lastConnID protocol.ConnectionID
	var processed bool
	data := rp.data
	p := rp
	for len(data) > 0 {
		if counter > 0 {
			p = p.Clone()
			p.data = data
		}

		hdr, packetData, rest, err := wire.ParsePacket(p.data, s.srcConnIDLen)
		if err != nil {
			if s.qlogger != nil {
				dropReason := qlog.PacketDropHeaderParseError
				if err == wire.ErrUnsupportedVersion {
					dropReason = qlog.PacketDropUnsupportedVersion
				}
				s.qlogger.DroppedPacket(qlog.PacketTypeNotDetermined, protocol.ByteCount(len(data)), dropReason)
			}
			s.logger.Debugf("error parsing packet: %s", err)
			break
		}

		if hdr.IsLongHeader && hdr.Version != s.version {
			if s.qlogger != nil {
				s.qlogger.DroppedPacket(qlog.PacketTypeFromHeader(hdr), protocol.ByteCount(len(data)), qlog.PacketDropUnexpectedVersion)
			}
			s.logger.Debugf("Dropping packet with version %x. Expected %x.", hdr.Version, s.version)
			break
		}

		if counter > 0 && !hdr.DestConnectionID.Equal(lastConnID) {
			if s.qlogger != nil {
				s.qlogger.DroppedPacket(qlog.PacketTypeFromHeader(hdr), protocol.ByteCount(len(data)), qlog.PacketDropUnknownConnectionID)
			}
			s.logger.Debugf("coalesced packet has different destination connection ID: %s, expected %s", hdr.DestConnectionID, lastConnID)
			break
		}
		lastConnID = hdr.DestConnectionID

		if counter > 0 {
			p.buffer.Split()
		}
		counter++

		// only log if this actually a coalesced packet
		if s.logger.Debug() && (counter > 1 || len(rest) > 0) {
			s.logger.Debugf("Parsed a coalesced packet. Part %d: %d bytes. Remaining: %d bytes.", counter, len(packetData), len(rest))
		}
		p.data = packetData
		if wasProcessed := s.handleSinglePacket(p, hdr); wasProcessed {
			processed = true
		}
		data = rest
	}
	p.buffer.MaybeRelease()
	return processed
}

func (s *session) handleSinglePacket(p *receivedPacket, hdr *wire.Header) bool /* was the packet successfully processed */ {
	var wasQueued bool

	defer func() {
		// Put back the packet buffer if the packet wasn't queued for later decryption.
		if !wasQueued {
			p.buffer.Decrement()
		}
	}()

	if hdr.Type == protocol.PacketTypeRetry {
		return s.handleRetryPacket(hdr, p.data)
	}

	// The server can change the source connection ID with the first Handshake packet.
	// After this, all packets with a different source connection have to be ignored.
	if s.receivedFirstPacket && hdr.IsLongHeader && !hdr.SrcConnectionID.Equal(s.handshakeDestConnID) {
		if s.qlogger != nil {
			s.qlogger.DroppedPacket(qlog.PacketTypeFromHeader(hdr), protocol.ByteCount(len(p.data)), qlog.PacketDropUnknownConnectionID)
		}
		s.logger.Debugf("Dropping %s packet (%d bytes) with unexpected source connection ID: %s (expected %s)", hdr.PacketType(), len(p.data), hdr.SrcConnectionID, s.handshakeDestConnID)
		return false
	}
	// drop 0-RTT packets, if we are a client
	if s.perspective == protocol.PerspectiveClient && hdr.Type == protocol.PacketType0RTT {
		s.qlogger.DroppedPacket(qlog.PacketType0RTT, protocol.ByteCount(len(p.data)), qlog.PacketDropKeyUnavailable)
		return false
	}

	packet, err := s.unpacker.Unpack(hdr, p.rcvTime, p.data)
	if err != nil {
		switch err {
		case handshake.ErrKeysDropped:
			if s.qlogger != nil {
				s.qlogger.DroppedPacket(qlog.PacketTypeFromHeader(hdr), protocol.ByteCount(len(p.data)), qlog.PacketDropKeyUnavailable)
			}
			s.logger.Debugf("Dropping %s packet (%d bytes) because we already dropped the keys.", hdr.PacketType(), len(p.data))
		case handshake.ErrKeysNotYetAvailable:
			// Sealer for this encryption level not yet available.
			// Try again later.
			wasQueued = true
			s.tryQueueingUndecryptablePacket(p, hdr)
		case wire.ErrInvalidReservedBits:
			s.closeLocal(qerr.NewError(qerr.ProtocolViolation, err.Error()))
		default:
			// This might be a packet injected by an attacker.
			// Drop it.
			if s.qlogger != nil {
				s.qlogger.DroppedPacket(qlog.PacketTypeFromHeader(hdr), protocol.ByteCount(len(p.data)), qlog.PacketDropPayloadDecryptError)
			}
			s.logger.Debugf("Dropping %s packet (%d bytes) that could not be unpacked. Error: %s", hdr.PacketType(), len(p.data), err)
		}
		return false
	}

	if s.logger.Debug() {
		s.logger.Debugf("<- Reading packet %d (%d bytes) for connection %s, %s", packet.packetNumber, len(p.data), hdr.DestConnectionID, packet.encryptionLevel)
		packet.hdr.Log(s.logger)
	}

	if err := s.handleUnpackedPacket(packet, p.rcvTime, protocol.ByteCount(len(p.data))); err != nil {
		s.closeLocal(err)
		return false
	}
	return true
}

func (s *session) handleRetryPacket(hdr *wire.Header, data []byte) bool /* was this a valid Retry */ {
	(&wire.ExtendedHeader{Header: *hdr}).Log(s.logger)
	if s.perspective == protocol.PerspectiveServer {
		if s.qlogger != nil {
			s.qlogger.DroppedPacket(qlog.PacketTypeRetry, protocol.ByteCount(len(data)), qlog.PacketDropUnexpectedPacket)
		}
		s.logger.Debugf("Ignoring Retry.")
		return false
	}
	if s.receivedFirstPacket {
		if s.qlogger != nil {
			s.qlogger.DroppedPacket(qlog.PacketTypeRetry, protocol.ByteCount(len(data)), qlog.PacketDropUnexpectedPacket)
		}
		s.logger.Debugf("Ignoring Retry, since we already received a packet.")
		return false
	}
	destConnID := s.connIDManager.Get()
	if hdr.SrcConnectionID.Equal(destConnID) {
		if s.qlogger != nil {
			s.qlogger.DroppedPacket(qlog.PacketTypeRetry, protocol.ByteCount(len(data)), qlog.PacketDropUnexpectedPacket)
		}
		s.logger.Debugf("Ignoring Retry, since the server didn't change the Source Connection ID.")
		return false
	}
	// If a token is already set, this means that we already received a Retry from the server.
	// Ignore this Retry packet.
	if s.receivedRetry {
		s.logger.Debugf("Ignoring Retry, since a Retry was already received.")
		return false
	}

	tag := handshake.GetRetryIntegrityTag(data[:len(data)-16], destConnID)
	if !bytes.Equal(data[len(data)-16:], tag[:]) {
		if s.qlogger != nil {
			s.qlogger.DroppedPacket(qlog.PacketTypeRetry, protocol.ByteCount(len(data)), qlog.PacketDropPayloadDecryptError)
		}
		s.logger.Debugf("Ignoring spoofed Retry. Integrity Tag doesn't match.")
		return false
	}

	s.logger.Debugf("<- Received Retry")
	s.logger.Debugf("Switching destination connection ID to: %s", hdr.SrcConnectionID)
	if s.qlogger != nil {
		s.qlogger.ReceivedRetry(hdr)
	}
	s.origDestConnID = s.handshakeDestConnID
	newDestConnID := hdr.SrcConnectionID
	s.receivedRetry = true
	if err := s.sentPacketHandler.ResetForRetry(); err != nil {
		s.closeLocal(err)
		return false
	}
	s.handshakeDestConnID = newDestConnID
	s.cryptoStreamHandler.ChangeConnectionID(newDestConnID)
	s.packer.SetToken(hdr.Token)
	s.connIDManager.ChangeInitialConnID(newDestConnID)
	s.scheduleSending()
	return true
}

func (s *session) handleUnpackedPacket(
	packet *unpackedPacket,
	rcvTime time.Time,
	packetSize protocol.ByteCount, // only for logging
) error {
	if len(packet.data) == 0 {
		return qerr.NewError(qerr.ProtocolViolation, "empty packet")
	}

	if !s.receivedFirstPacket {
		s.receivedFirstPacket = true
		// The server can change the source connection ID with the first Handshake packet.
		if s.perspective == protocol.PerspectiveClient && packet.hdr.IsLongHeader && !packet.hdr.SrcConnectionID.Equal(s.handshakeDestConnID) {
			cid := packet.hdr.SrcConnectionID
			s.logger.Debugf("Received first packet. Switching destination connection ID to: %s", cid)
			s.handshakeDestConnID = cid
			s.connIDManager.ChangeInitialConnID(cid)
		}
		// We create the session as soon as we receive the first packet from the client.
		// We do that before authenticating the packet.
		// That means that if the source connection ID was corrupted,
		// we might have create a session with an incorrect source connection ID.
		// Once we authenticate the first packet, we need to update it.
		if s.perspective == protocol.PerspectiveServer {
			if !packet.hdr.SrcConnectionID.Equal(s.handshakeDestConnID) {
				s.handshakeDestConnID = packet.hdr.SrcConnectionID
				s.connIDManager.ChangeInitialConnID(packet.hdr.SrcConnectionID)
			}
			if s.qlogger != nil {
				s.qlogger.StartedConnection(
					s.conn.LocalAddr(),
					s.conn.RemoteAddr(),
					s.version,
					packet.hdr.SrcConnectionID,
					packet.hdr.DestConnectionID,
				)
			}
		}
	}

	s.lastPacketReceivedTime = rcvTime
	s.firstAckElicitingPacketAfterIdleSentTime = time.Time{}
	s.keepAlivePingSent = false

	// Only used for tracing.
	// If we're not tracing, this slice will always remain empty.
	var frames []wire.Frame
	var transportState *quictrace.TransportState

	r := bytes.NewReader(packet.data)
	var isAckEliciting bool
	for {
		frame, err := s.frameParser.ParseNext(r, packet.encryptionLevel)
		if err != nil {
			return err
		}
		if frame == nil {
			break
		}
		if ackhandler.IsFrameAckEliciting(frame) {
			isAckEliciting = true
		}
		if s.traceCallback != nil || s.qlogger != nil {
			frames = append(frames, frame)
		}
		// Only process frames now if we're not logging.
		// If we're logging, we need to make sure that the packet_received event is logged first.
		if s.qlogger == nil {
			if err := s.handleFrame(frame, packet.encryptionLevel); err != nil {
				return err
			}
		}
	}

	if s.traceCallback != nil {
		transportState = s.sentPacketHandler.GetStats()
		s.traceCallback(quictrace.Event{
			Time:            rcvTime,
			EventType:       quictrace.PacketReceived,
			TransportState:  transportState,
			EncryptionLevel: packet.encryptionLevel,
			PacketNumber:    packet.packetNumber,
			PacketSize:      protocol.ByteCount(len(packet.data)),
			Frames:          frames,
		})
	}
	if s.qlogger != nil {
		s.qlogger.ReceivedPacket(packet.hdr, packetSize, frames)
		for _, frame := range frames {
			if err := s.handleFrame(frame, packet.encryptionLevel); err != nil {
				return err
			}
		}
	}

	return s.receivedPacketHandler.ReceivedPacket(packet.packetNumber, packet.encryptionLevel, rcvTime, isAckEliciting)
}

func (s *session) handleFrame(f wire.Frame, encLevel protocol.EncryptionLevel) error {
	var err error
	wire.LogFrame(s.logger, f, false)
	switch frame := f.(type) {
	case *wire.CryptoFrame:
		err = s.handleCryptoFrame(frame, encLevel)
	case *wire.StreamFrame:
		err = s.handleStreamFrame(frame)
	case *wire.AckFrame:
		err = s.handleAckFrame(frame, encLevel)
	case *wire.ConnectionCloseFrame:
		s.handleConnectionCloseFrame(frame)
	case *wire.ResetStreamFrame:
		err = s.handleResetStreamFrame(frame)
	case *wire.MaxDataFrame:
		s.handleMaxDataFrame(frame)
	case *wire.MaxStreamDataFrame:
		err = s.handleMaxStreamDataFrame(frame)
	case *wire.MaxStreamsFrame:
		err = s.handleMaxStreamsFrame(frame)
	case *wire.DataBlockedFrame:
	case *wire.StreamDataBlockedFrame:
	case *wire.StreamsBlockedFrame:
	case *wire.StopSendingFrame:
		err = s.handleStopSendingFrame(frame)
	case *wire.PingFrame:
	case *wire.PathChallengeFrame:
		s.handlePathChallengeFrame(frame)
	case *wire.PathResponseFrame:
		// since we don't send PATH_CHALLENGEs, we don't expect PATH_RESPONSEs
		err = errors.New("unexpected PATH_RESPONSE frame")
	case *wire.NewTokenFrame:
		err = s.handleNewTokenFrame(frame)
	case *wire.NewConnectionIDFrame:
		err = s.handleNewConnectionIDFrame(frame)
	case *wire.RetireConnectionIDFrame:
		err = s.handleRetireConnectionIDFrame(frame)
	case *wire.HandshakeDoneFrame:
		err = s.handleHandshakeDoneFrame()
	default:
		err = fmt.Errorf("unexpected frame type: %s", reflect.ValueOf(&frame).Elem().Type().Name())
	}
	return err
}

// handlePacket is called by the server with a new packet
func (s *session) handlePacket(p *receivedPacket) {
	// Discard packets once the amount of queued packets is larger than
	// the channel size, protocol.MaxSessionUnprocessedPackets
	select {
	case s.receivedPackets <- p:
	default:
	}
}

func (s *session) handleConnectionCloseFrame(frame *wire.ConnectionCloseFrame) {
	var e error
	if frame.IsApplicationError {
		e = qerr.NewApplicationError(frame.ErrorCode, frame.ReasonPhrase)
	} else {
		e = qerr.NewError(frame.ErrorCode, frame.ReasonPhrase)
	}
	s.closeRemote(e)
}

func (s *session) handleCryptoFrame(frame *wire.CryptoFrame, encLevel protocol.EncryptionLevel) error {
	encLevelChanged, err := s.cryptoStreamManager.HandleCryptoFrame(frame, encLevel)
	if err != nil {
		return err
	}
	if encLevelChanged {
		s.tryDecryptingQueuedPackets()
	}
	return nil
}

func (s *session) handleStreamFrame(frame *wire.StreamFrame) error {
	str, err := s.streamsMap.GetOrOpenReceiveStream(frame.StreamID)
	if err != nil {
		return err
	}
	if str == nil {
		// Stream is closed and already garbage collected
		// ignore this StreamFrame
		return nil
	}
	return str.handleStreamFrame(frame)
}

func (s *session) handleMaxDataFrame(frame *wire.MaxDataFrame) {
	s.connFlowController.UpdateSendWindow(frame.ByteOffset)
}

func (s *session) handleMaxStreamDataFrame(frame *wire.MaxStreamDataFrame) error {
	str, err := s.streamsMap.GetOrOpenSendStream(frame.StreamID)
	if err != nil {
		return err
	}
	if str == nil {
		// stream is closed and already garbage collected
		return nil
	}
	str.handleMaxStreamDataFrame(frame)
	return nil
}

func (s *session) handleMaxStreamsFrame(frame *wire.MaxStreamsFrame) error {
	return s.streamsMap.HandleMaxStreamsFrame(frame)
}

func (s *session) handleResetStreamFrame(frame *wire.ResetStreamFrame) error {
	str, err := s.streamsMap.GetOrOpenReceiveStream(frame.StreamID)
	if err != nil {
		return err
	}
	if str == nil {
		// stream is closed and already garbage collected
		return nil
	}
	return str.handleResetStreamFrame(frame)
}

func (s *session) handleStopSendingFrame(frame *wire.StopSendingFrame) error {
	str, err := s.streamsMap.GetOrOpenSendStream(frame.StreamID)
	if err != nil {
		return err
	}
	if str == nil {
		// stream is closed and already garbage collected
		return nil
	}
	str.handleStopSendingFrame(frame)
	return nil
}

func (s *session) handlePathChallengeFrame(frame *wire.PathChallengeFrame) {
	s.queueControlFrame(&wire.PathResponseFrame{Data: frame.Data})
}

func (s *session) handleNewTokenFrame(frame *wire.NewTokenFrame) error {
	if s.perspective == protocol.PerspectiveServer {
		return qerr.NewError(qerr.ProtocolViolation, "Received NEW_TOKEN frame from the client.")
	}
	if s.config.TokenStore != nil {
		s.config.TokenStore.Put(s.tokenStoreKey, &ClientToken{data: frame.Token})
	}
	return nil
}

func (s *session) handleNewConnectionIDFrame(f *wire.NewConnectionIDFrame) error {
	return s.connIDManager.Add(f)
}

func (s *session) handleRetireConnectionIDFrame(f *wire.RetireConnectionIDFrame) error {
	return s.connIDGenerator.Retire(f.SequenceNumber)
}

func (s *session) handleHandshakeDoneFrame() error {
	if s.perspective == protocol.PerspectiveServer {
		return qerr.NewError(qerr.ProtocolViolation, "received a HANDSHAKE_DONE frame")
	}
	s.cryptoStreamHandler.DropHandshakeKeys()
	return nil
}

func (s *session) handleAckFrame(frame *wire.AckFrame, encLevel protocol.EncryptionLevel) error {
	if err := s.sentPacketHandler.ReceivedAck(frame, encLevel, s.lastPacketReceivedTime); err != nil {
		return err
	}
	if encLevel == protocol.Encryption1RTT {
		s.cryptoStreamHandler.SetLargest1RTTAcked(frame.LargestAcked())
	}
	return nil
}

// closeLocal closes the session and send a CONNECTION_CLOSE containing the error
func (s *session) closeLocal(e error) {
	s.closeOnce.Do(func() {
		if e == nil {
			s.logger.Infof("Closing session.")
		} else {
			s.logger.Errorf("Closing session with error: %s", e)
		}
		s.closeChan <- closeError{err: e, immediate: false, remote: false}
	})
}

// destroy closes the session without sending the error on the wire
func (s *session) destroy(e error) {
	s.destroyImpl(e)
	<-s.ctx.Done()
}

func (s *session) destroyImpl(e error) {
	s.closeOnce.Do(func() {
		if nerr, ok := e.(net.Error); ok && nerr.Timeout() {
			s.logger.Errorf("Destroying session: %s", e)
		} else {
			s.logger.Errorf("Destroying session with error: %s", e)
		}
		s.closeChan <- closeError{err: e, immediate: true, remote: false}
	})
}

// closeForRecreating closes the session in order to recreate it immediately afterwards
// It returns the first packet number that should be used in the new session.
func (s *session) closeForRecreating() protocol.PacketNumber {
	s.destroy(errCloseForRecreating)
	nextPN, _ := s.sentPacketHandler.PeekPacketNumber(protocol.EncryptionInitial)
	return nextPN
}

func (s *session) closeRemote(e error) {
	s.closeOnce.Do(func() {
		s.logger.Errorf("Peer closed session with error: %s", e)
		s.closeChan <- closeError{err: e, immediate: true, remote: true}
	})
}

// Close the connection. It sends a NO_ERROR transport error.
// It waits until the run loop has stopped before returning
func (s *session) shutdown() {
	s.closeLocal(nil)
	<-s.ctx.Done()
}

func (s *session) CloseWithError(code protocol.ApplicationErrorCode, desc string) error {
	s.closeLocal(qerr.NewApplicationError(qerr.ErrorCode(code), desc))
	<-s.ctx.Done()
	return nil
}

func (s *session) handleCloseError(closeErr closeError) {
	if closeErr.err == nil {
		closeErr.err = qerr.NewApplicationError(0, "")
	}
	if statelessReset, ok := closeErr.err.(interface{ StatelessResetToken() *[16]byte }); ok && s.qlogger != nil {
		s.qlogger.ReceivedStatelessReset(statelessReset.StatelessResetToken())
	}

	var quicErr *qerr.QuicError
	var ok bool
	if quicErr, ok = closeErr.err.(*qerr.QuicError); !ok {
		quicErr = qerr.ToQuicError(closeErr.err)
	}

	s.streamsMap.CloseWithError(quicErr)
	s.connIDManager.Close()

	// If this is a remote close we're done here
	if closeErr.remote {
		s.connIDGenerator.ReplaceWithClosed(newClosedRemoteSession(s.perspective))
		return
	}
	if closeErr.immediate {
		s.connIDGenerator.RemoveAll()
		return
	}
	connClosePacket, err := s.sendConnectionClose(quicErr)
	if err != nil {
		s.logger.Debugf("Error sending CONNECTION_CLOSE: %s", err)
	}
	cs := newClosedLocalSession(s.conn, connClosePacket, s.perspective, s.logger)
	s.connIDGenerator.ReplaceWithClosed(cs)
}

func (s *session) dropEncryptionLevel(encLevel protocol.EncryptionLevel) {
	if encLevel == protocol.EncryptionHandshake {
		s.handshakeConfirmed = true
	}
	s.sentPacketHandler.DropPackets(encLevel)
	s.receivedPacketHandler.DropPackets(encLevel)
	if s.qlogger != nil {
		s.qlogger.DroppedEncryptionLevel(encLevel)
	}
}

func (s *session) processTransportParameters(params *wire.TransportParameters) {
	if s.logger.Debug() {
		s.logger.Debugf("Processed Transport Parameters: %s", params)
	}
	if s.qlogger != nil {
		s.qlogger.ReceivedTransportParameters(params)
	}

	// check the Retry token
	if s.perspective == protocol.PerspectiveClient && !params.OriginalConnectionID.Equal(s.origDestConnID) {
		s.closeLocal(qerr.NewError(qerr.TransportParameterError, fmt.Sprintf("expected original_connection_id to equal %s, is %s", s.origDestConnID, params.OriginalConnectionID)))
		return
	}

	s.peerParams = params
	// Our local idle timeout will always be > 0.
	s.idleTimeout = utils.MinNonZeroDuration(s.config.MaxIdleTimeout, params.MaxIdleTimeout)
	s.keepAliveInterval = utils.MinDuration(s.idleTimeout/2, protocol.MaxKeepAliveInterval)
	if err := s.streamsMap.UpdateLimits(params); err != nil {
		s.closeLocal(err)
		return
	}
	s.packer.HandleTransportParameters(params)
	s.frameParser.SetAckDelayExponent(params.AckDelayExponent)
	s.connFlowController.UpdateSendWindow(params.InitialMaxData)
	s.rttStats.SetMaxAckDelay(params.MaxAckDelay)
	s.connIDGenerator.SetMaxActiveConnIDs(params.ActiveConnectionIDLimit)
	if params.StatelessResetToken != nil {
		s.connIDManager.SetStatelessResetToken(*params.StatelessResetToken)
	}
	// We don't support connection migration yet, so we don't have any use for the preferred_address.
	if params.PreferredAddress != nil {
		s.logger.Debugf("Server sent preferred_address. Retiring the preferred_address connection ID.")
		// Retire the connection ID.
		s.connIDManager.AddFromPreferredAddress(params.PreferredAddress.ConnectionID, &params.PreferredAddress.StatelessResetToken)
	}
	// On the server side, the early session is ready as soon as we processed
	// the client's transport parameters.
	if s.perspective == protocol.PerspectiveServer {
		close(s.earlySessionReadyChan)
	}
}

func (s *session) sendPackets() error {
	s.pacingDeadline = time.Time{}

	sendMode := s.sentPacketHandler.SendMode()
	if sendMode == ackhandler.SendNone { // shortcut: return immediately if there's nothing to send
		return nil
	}

	numPackets := s.sentPacketHandler.ShouldSendNumPackets()
	var numPacketsSent int
sendLoop:
	for {
		switch sendMode {
		case ackhandler.SendNone:
			break sendLoop
		case ackhandler.SendAck:
			// If we already sent packets, and the send mode switches to SendAck,
			// we've just become congestion limited.
			// There's no need to try to send an ACK at this moment.
			if numPacketsSent > 0 {
				return nil
			}
			// We can at most send a single ACK only packet.
			// There will only be a new ACK after receiving new packets.
			// SendAck is only returned when we're congestion limited, so we don't need to set the pacingt timer.
			return s.maybeSendAckOnlyPacket()
		case ackhandler.SendPTOInitial:
			if err := s.sendProbePacket(protocol.EncryptionInitial); err != nil {
				return err
			}
			numPacketsSent++
		case ackhandler.SendPTOHandshake:
			if err := s.sendProbePacket(protocol.EncryptionHandshake); err != nil {
				return err
			}
			numPacketsSent++
		case ackhandler.SendPTOAppData:
			if err := s.sendProbePacket(protocol.Encryption1RTT); err != nil {
				return err
			}
			numPacketsSent++
		case ackhandler.SendAny:
			sentPacket, err := s.sendPacket()
			if err != nil {
				return err
			}
			if !sentPacket {
				break sendLoop
			}
			numPacketsSent++
		default:
			return fmt.Errorf("BUG: invalid send mode %d", sendMode)
		}
		if numPacketsSent >= numPackets {
			break
		}
		sendMode = s.sentPacketHandler.SendMode()
	}
	// Only start the pacing timer if we sent as many packets as we were allowed.
	// There will probably be more to send when calling sendPacket again.
	if numPacketsSent == numPackets {
		s.pacingDeadline = s.sentPacketHandler.TimeUntilSend()
	}
	return nil
}

func (s *session) maybeSendAckOnlyPacket() error {
	packet, err := s.packer.MaybePackAckPacket(s.handshakeConfirmed)
	if err != nil {
		return err
	}
	if packet == nil {
		return nil
	}
	s.sendPackedPacket(packet)
	return nil
}

func (s *session) sendProbePacket(encLevel protocol.EncryptionLevel) error {
	// Queue probe packets until we actually send out a packet,
	// or until there are no more packets to queue.
	var packet *packedPacket
	for {
		if wasQueued := s.sentPacketHandler.QueueProbePacket(encLevel); !wasQueued {
			break
		}
		var err error
		packet, err = s.packer.MaybePackProbePacket(encLevel)
		if err != nil {
			return err
		}
		if packet != nil {
			break
		}
	}
	if packet == nil {
		switch encLevel {
		case protocol.EncryptionInitial:
			s.retransmissionQueue.AddInitial(&wire.PingFrame{})
		case protocol.EncryptionHandshake:
			s.retransmissionQueue.AddHandshake(&wire.PingFrame{})
		case protocol.Encryption1RTT:
			s.retransmissionQueue.AddAppData(&wire.PingFrame{})
		default:
			panic("unexpected encryption level")
		}
		var err error
		packet, err = s.packer.MaybePackProbePacket(encLevel)
		if err != nil {
			return err
		}
	}
	if packet == nil || packet.packetContents == nil {
		return fmt.Errorf("session BUG: couldn't pack %s probe packet", encLevel)
	}
	s.sendPackedPacket(packet)
	return nil
}

func (s *session) sendPacket() (bool, error) {
	if isBlocked, offset := s.connFlowController.IsNewlyBlocked(); isBlocked {
		s.framer.QueueControlFrame(&wire.DataBlockedFrame{DataLimit: offset})
	}
	s.windowUpdateQueue.QueueAll()

	if !s.handshakeConfirmed {
		now := time.Now()
		packet, err := s.packer.PackCoalescedPacket()
		if err != nil || packet == nil {
			return false, err
		}
		s.logCoalescedPacket(now, packet)
		for _, p := range packet.packets {
			if s.firstAckElicitingPacketAfterIdleSentTime.IsZero() && p.IsAckEliciting() {
				s.firstAckElicitingPacketAfterIdleSentTime = now
			}
			s.sentPacketHandler.SentPacket(p.ToAckHandlerPacket(now, s.retransmissionQueue))
		}
		s.connIDManager.SentPacket()
		s.sendQueue.Send(packet.buffer)
		return true, nil
	}
	packet, err := s.packer.PackPacket()
	if err != nil || packet == nil {
		return false, err
	}
	s.sendPackedPacket(packet)
	return true, nil
}

func (s *session) sendPackedPacket(packet *packedPacket) {
	now := time.Now()
	if s.firstAckElicitingPacketAfterIdleSentTime.IsZero() && packet.IsAckEliciting() {
		s.firstAckElicitingPacketAfterIdleSentTime = now
	}
	s.logPacket(now, packet)
	s.sentPacketHandler.SentPacket(packet.ToAckHandlerPacket(time.Now(), s.retransmissionQueue))
	s.connIDManager.SentPacket()
	s.sendQueue.Send(packet.buffer)
}

func (s *session) sendConnectionClose(quicErr *qerr.QuicError) ([]byte, error) {
	packet, err := s.packer.PackConnectionClose(quicErr)
	if err != nil {
		return nil, err
	}
	s.logCoalescedPacket(time.Now(), packet)
	return packet.buffer.Data, s.conn.Write(packet.buffer.Data)
}

func (s *session) logPacketContents(now time.Time, p *packetContents) {
	// qlog
	if s.qlogger != nil {
		frames := make([]wire.Frame, 0, len(p.frames))
		for _, f := range p.frames {
			frames = append(frames, f.Frame)
		}
		s.qlogger.SentPacket(p.header, p.length, p.ack, frames)
	}

	// quic-trace
	if s.traceCallback != nil {
		frames := make([]wire.Frame, 0, len(p.frames))
		for _, f := range p.frames {
			frames = append(frames, f.Frame)
		}
		s.traceCallback(quictrace.Event{
			Time:            now,
			EventType:       quictrace.PacketSent,
			TransportState:  s.sentPacketHandler.GetStats(),
			EncryptionLevel: p.EncryptionLevel(),
			PacketNumber:    p.header.PacketNumber,
			PacketSize:      p.length,
			Frames:          frames,
		})
	}

	// quic-go logging
	if !s.logger.Debug() {
		return
	}
	p.header.Log(s.logger)
	if p.ack != nil {
		wire.LogFrame(s.logger, p.ack, true)
	}
	for _, frame := range p.frames {
		wire.LogFrame(s.logger, frame.Frame, true)
	}
}

func (s *session) logCoalescedPacket(now time.Time, packet *coalescedPacket) {
	if s.logger.Debug() {
		if len(packet.packets) > 1 {
			s.logger.Debugf("-> Sending coalesced packet (%d parts, %d bytes) for connection %s", len(packet.packets), packet.buffer.Len(), s.logID)
		} else {
			s.logger.Debugf("-> Sending packet %d (%d bytes) for connection %s, %s", packet.packets[0].header.PacketNumber, packet.buffer.Len(), s.logID, packet.packets[0].EncryptionLevel())
		}
	}
	for _, p := range packet.packets {
		s.logPacketContents(now, p)
	}
}

func (s *session) logPacket(now time.Time, packet *packedPacket) {
	if s.logger.Debug() {
		s.logger.Debugf("-> Sending packet %d (%d bytes) for connection %s, %s", packet.header.PacketNumber, packet.buffer.Len(), s.logID, packet.EncryptionLevel())
	}
	s.logPacketContents(now, packet.packetContents)
}

// AcceptStream returns the next stream openend by the peer
func (s *session) AcceptStream(ctx context.Context) (Stream, error) {
	return s.streamsMap.AcceptStream(ctx)
}

func (s *session) AcceptUniStream(ctx context.Context) (ReceiveStream, error) {
	return s.streamsMap.AcceptUniStream(ctx)
}

// OpenStream opens a stream
func (s *session) OpenStream() (Stream, error) {
	return s.streamsMap.OpenStream()
}

func (s *session) OpenStreamSync(ctx context.Context) (Stream, error) {
	return s.streamsMap.OpenStreamSync(ctx)
}

func (s *session) OpenUniStream() (SendStream, error) {
	return s.streamsMap.OpenUniStream()
}

func (s *session) OpenUniStreamSync(ctx context.Context) (SendStream, error) {
	return s.streamsMap.OpenUniStreamSync(ctx)
}

func (s *session) newFlowController(id protocol.StreamID) flowcontrol.StreamFlowController {
	var initialSendWindow protocol.ByteCount
	if s.peerParams != nil {
		if id.Type() == protocol.StreamTypeUni {
			initialSendWindow = s.peerParams.InitialMaxStreamDataUni
		} else {
			if id.InitiatedBy() == s.perspective {
				initialSendWindow = s.peerParams.InitialMaxStreamDataBidiRemote
			} else {
				initialSendWindow = s.peerParams.InitialMaxStreamDataBidiLocal
			}
		}
	}
	return flowcontrol.NewStreamFlowController(
		id,
		s.connFlowController,
		protocol.InitialMaxStreamData,
		protocol.ByteCount(s.config.MaxReceiveStreamFlowControlWindow),
		initialSendWindow,
		s.onHasStreamWindowUpdate,
		s.rttStats,
		s.logger,
	)
}

// scheduleSending signals that we have data for sending
func (s *session) scheduleSending() {
	select {
	case s.sendingScheduled <- struct{}{}:
	default:
	}
}

func (s *session) tryQueueingUndecryptablePacket(p *receivedPacket, hdr *wire.Header) {
	if len(s.undecryptablePackets)+1 > protocol.MaxUndecryptablePackets {
		if s.qlogger != nil {
			s.qlogger.DroppedPacket(qlog.PacketTypeFromHeader(hdr), protocol.ByteCount(len(p.data)), qlog.PacketDropDOSPrevention)
		}
		s.logger.Infof("Dropping undecryptable packet (%d bytes). Undecryptable packet queue full.", len(p.data))
		return
	}
	s.logger.Infof("Queueing packet (%d bytes) for later decryption", len(p.data))
	if s.qlogger != nil {
		s.qlogger.BufferedPacket(qlog.PacketTypeFromHeader(hdr))
	}
	s.undecryptablePackets = append(s.undecryptablePackets, p)
}

func (s *session) tryDecryptingQueuedPackets() {
	for _, p := range s.undecryptablePackets {
		s.handlePacket(p)
	}
	s.undecryptablePackets = s.undecryptablePackets[:0]
}

func (s *session) queueControlFrame(f wire.Frame) {
	s.framer.QueueControlFrame(f)
	s.scheduleSending()
}

func (s *session) onHasStreamWindowUpdate(id protocol.StreamID) {
	s.windowUpdateQueue.AddStream(id)
	s.scheduleSending()
}

func (s *session) onHasConnectionWindowUpdate() {
	s.windowUpdateQueue.AddConnection()
	s.scheduleSending()
}

func (s *session) onHasStreamData(id protocol.StreamID) {
	s.framer.AddActiveStream(id)
	s.scheduleSending()
}

func (s *session) onStreamCompleted(id protocol.StreamID) {
	if err := s.streamsMap.DeleteStream(id); err != nil {
		s.closeLocal(err)
	}
}

func (s *session) LocalAddr() net.Addr {
	return s.conn.LocalAddr()
}

func (s *session) RemoteAddr() net.Addr {
	return s.conn.RemoteAddr()
}

func (s *session) getPerspective() protocol.Perspective {
	return s.perspective
}

func (s *session) GetVersion() protocol.VersionNumber {
	return s.version
}
