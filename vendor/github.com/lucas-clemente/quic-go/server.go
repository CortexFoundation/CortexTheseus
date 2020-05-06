package quic

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
	"time"

	"github.com/lucas-clemente/quic-go/qlog"

	"github.com/lucas-clemente/quic-go/internal/handshake"
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/qerr"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

// packetHandler handles packets
type packetHandler interface {
	handlePacket(*receivedPacket)
	shutdown()
	destroy(error)
	getPerspective() protocol.Perspective
}

type unknownPacketHandler interface {
	handlePacket(*receivedPacket)
	setCloseError(error)
}

type packetHandlerManager interface {
	Destroy() error
	sessionRunner
	SetServer(unknownPacketHandler)
	CloseServer()
}

type quicSession interface {
	EarlySession
	earlySessionReady() <-chan struct{}
	handlePacket(*receivedPacket)
	GetVersion() protocol.VersionNumber
	getPerspective() protocol.Perspective
	run() error
	destroy(error)
	shutdown()
	closeForRecreating() protocol.PacketNumber
}

// A Listener of QUIC
type baseServer struct {
	mutex sync.Mutex

	acceptEarlySessions bool

	tlsConf *tls.Config
	config  *Config

	conn net.PacketConn
	// If the server is started with ListenAddr, we create a packet conn.
	// If it is started with Listen, we take a packet conn as a parameter.
	createdPacketConn bool

	tokenGenerator *handshake.TokenGenerator

	zeroRTTQueue   *zeroRTTQueue
	sessionHandler packetHandlerManager

	receivedPackets chan *receivedPacket

	// set as a member, so they can be set in the tests
	newSession func(connection, sessionRunner, protocol.ConnectionID /* original connection ID */, protocol.ConnectionID /* client dest connection ID */, protocol.ConnectionID /* destination connection ID */, protocol.ConnectionID /* source connection ID */, [16]byte, *Config, *tls.Config, *handshake.TokenGenerator, bool /* enable 0-RTT */, qlog.Tracer, utils.Logger, protocol.VersionNumber) quicSession

	serverError error
	errorChan   chan struct{}
	closed      bool

	sessionQueue    chan quicSession
	sessionQueueLen int32 // to be used as an atomic

	logger utils.Logger
}

var _ Listener = &baseServer{}
var _ unknownPacketHandler = &baseServer{}

type earlyServer struct{ *baseServer }

var _ EarlyListener = &earlyServer{}

func (s *earlyServer) Accept(ctx context.Context) (EarlySession, error) {
	return s.baseServer.accept(ctx)
}

// ListenAddr creates a QUIC server listening on a given address.
// The tls.Config must not be nil and must contain a certificate configuration.
// The quic.Config may be nil, in that case the default values will be used.
func ListenAddr(addr string, tlsConf *tls.Config, config *Config) (Listener, error) {
	return listenAddr(addr, tlsConf, config, false)
}

// ListenAddrEarly works like ListenAddr, but it returns sessions before the handshake completes.
func ListenAddrEarly(addr string, tlsConf *tls.Config, config *Config) (EarlyListener, error) {
	s, err := listenAddr(addr, tlsConf, config, true)
	if err != nil {
		return nil, err
	}
	return &earlyServer{s}, nil
}

func listenAddr(addr string, tlsConf *tls.Config, config *Config, acceptEarly bool) (*baseServer, error) {
	udpAddr, err := net.ResolveUDPAddr("udp", addr)
	if err != nil {
		return nil, err
	}
	conn, err := net.ListenUDP("udp", udpAddr)
	if err != nil {
		return nil, err
	}
	serv, err := listen(conn, tlsConf, config, acceptEarly)
	if err != nil {
		return nil, err
	}
	serv.createdPacketConn = true
	return serv, nil
}

// Listen listens for QUIC connections on a given net.PacketConn.
// A single net.PacketConn only be used for a single call to Listen.
// The PacketConn can be used for simultaneous calls to Dial.
// QUIC connection IDs are used for demultiplexing the different connections.
// The tls.Config must not be nil and must contain a certificate configuration.
// The tls.Config.CipherSuites allows setting of TLS 1.3 cipher suites.
// Furthermore, it must define an application control (using NextProtos).
// The quic.Config may be nil, in that case the default values will be used.
func Listen(conn net.PacketConn, tlsConf *tls.Config, config *Config) (Listener, error) {
	return listen(conn, tlsConf, config, false)
}

// ListenEarly works like Listen, but it returns sessions before the handshake completes.
func ListenEarly(conn net.PacketConn, tlsConf *tls.Config, config *Config) (EarlyListener, error) {
	s, err := listen(conn, tlsConf, config, true)
	if err != nil {
		return nil, err
	}
	return &earlyServer{s}, nil
}

func listen(conn net.PacketConn, tlsConf *tls.Config, config *Config, acceptEarly bool) (*baseServer, error) {
	if tlsConf == nil {
		return nil, errors.New("quic: tls.Config not set")
	}
	config = populateServerConfig(config)
	for _, v := range config.Versions {
		if !protocol.IsValidVersion(v) {
			return nil, fmt.Errorf("%s is not a valid QUIC version", v)
		}
	}

	sessionHandler, err := getMultiplexer().AddConn(conn, config.ConnectionIDLength, config.StatelessResetKey)
	if err != nil {
		return nil, err
	}
	tokenGenerator, err := handshake.NewTokenGenerator()
	if err != nil {
		return nil, err
	}
	s := &baseServer{
		conn:                conn,
		tlsConf:             tlsConf,
		config:              config,
		tokenGenerator:      tokenGenerator,
		sessionHandler:      sessionHandler,
		zeroRTTQueue:        newZeroRTTQueue(),
		sessionQueue:        make(chan quicSession),
		errorChan:           make(chan struct{}),
		receivedPackets:     make(chan *receivedPacket, protocol.MaxServerUnprocessedPackets),
		newSession:          newSession,
		logger:              utils.DefaultLogger.WithPrefix("server"),
		acceptEarlySessions: acceptEarly,
	}
	go s.run()
	sessionHandler.SetServer(s)
	s.logger.Debugf("Listening for %s connections on %s", conn.LocalAddr().Network(), conn.LocalAddr().String())
	return s, nil
}

func (s *baseServer) run() {
	for {
		select {
		case <-s.errorChan:
			return
		default:
		}
		select {
		case <-s.errorChan:
			return
		case p := <-s.receivedPackets:
			if shouldReleaseBuffer := s.handlePacketImpl(p); !shouldReleaseBuffer {
				p.buffer.Release()
			}
		}
	}
}

var defaultAcceptToken = func(clientAddr net.Addr, token *Token) bool {
	if token == nil {
		return false
	}
	validity := protocol.TokenValidity
	if token.IsRetryToken {
		validity = protocol.RetryTokenValidity
	}
	if time.Now().After(token.SentTime.Add(validity)) {
		return false
	}
	var sourceAddr string
	if udpAddr, ok := clientAddr.(*net.UDPAddr); ok {
		sourceAddr = udpAddr.IP.String()
	} else {
		sourceAddr = clientAddr.String()
	}
	return sourceAddr == token.RemoteAddr
}

// Accept returns sessions that already completed the handshake.
// It is only valid if acceptEarlySessions is false.
func (s *baseServer) Accept(ctx context.Context) (Session, error) {
	return s.accept(ctx)
}

func (s *baseServer) accept(ctx context.Context) (quicSession, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case sess := <-s.sessionQueue:
		atomic.AddInt32(&s.sessionQueueLen, -1)
		return sess, nil
	case <-s.errorChan:
		return nil, s.serverError
	}
}

// Close the server
func (s *baseServer) Close() error {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return nil
	}
	s.sessionHandler.CloseServer()
	if s.serverError == nil {
		s.serverError = errors.New("server closed")
	}
	var err error
	// If the server was started with ListenAddr, we created the packet conn.
	// We need to close it in order to make the go routine reading from that conn return.
	if s.createdPacketConn {
		err = s.sessionHandler.Destroy()
	}
	s.closed = true
	close(s.errorChan)
	return err
}

func (s *baseServer) setCloseError(e error) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	s.serverError = e
	close(s.errorChan)
}

// Addr returns the server's network address
func (s *baseServer) Addr() net.Addr {
	return s.conn.LocalAddr()
}

func (s *baseServer) handlePacket(p *receivedPacket) {
	select {
	case s.receivedPackets <- p:
	default:
		s.logger.Debugf("Dropping packet from %s (%d bytes). Server receive queue full.", p.remoteAddr, len(p.data))
	}
}

func (s *baseServer) handlePacketImpl(p *receivedPacket) bool /* should the buffer be released */ {
	// If we're creating a new session, the packet will be passed to the session.
	// The header will then be parsed again.
	hdr, _, _, err := wire.ParsePacket(p.data, s.config.ConnectionIDLength)
	if err != nil && err != wire.ErrUnsupportedVersion {
		s.logger.Debugf("Error parsing packet: %s", err)
		return false
	}
	// Short header packets should never end up here in the first place
	if !hdr.IsLongHeader {
		return false
	}
	if hdr.Type == protocol.PacketTypeInitial && len(p.data) < protocol.MinInitialPacketSize {
		s.logger.Debugf("Dropping a packet that is too small to be a valid Initial (%d bytes)", len(p.data))
		return false
	}
	// send a Version Negotiation Packet if the client is speaking a different protocol version
	if !protocol.IsSupportedVersion(s.config.Versions, hdr.Version) {
		go s.sendVersionNegotiationPacket(p, hdr)
		return false
	}
	if hdr.IsLongHeader {
		if hdr.Type == protocol.PacketType0RTT {
			s.zeroRTTQueue.Enqueue(hdr.DestConnectionID, p)
			return true
		} else if hdr.Type != protocol.PacketTypeInitial {
			// Drop long header packets.
			// There's litte point in sending a Stateless Reset, since the client
			// might not have received the token yet.
			s.logger.Debugf("Dropping long header packet of type %s (%d bytes)", hdr.Type, len(p.data))
			return false
		}
	}

	s.logger.Debugf("<- Received Initial packet.")

	if err := s.handleInitialImpl(p, hdr); err != nil {
		s.logger.Errorf("Error occurred handling initial packet: %s", err)
	}
	// Don't put the packet buffer back.
	// handleInitialImpl deals with the buffer.
	return true
}

func (s *baseServer) handleInitialImpl(p *receivedPacket, hdr *wire.Header) error {
	if len(hdr.Token) == 0 && hdr.DestConnectionID.Len() < protocol.MinConnectionIDLenInitial {
		p.buffer.Release()
		return errors.New("too short connection ID")
	}

	var token *Token
	var origDestConnectionID protocol.ConnectionID
	if len(hdr.Token) > 0 {
		c, err := s.tokenGenerator.DecodeToken(hdr.Token)
		if err == nil {
			token = &Token{
				IsRetryToken: c.IsRetryToken,
				RemoteAddr:   c.RemoteAddr,
				SentTime:     c.SentTime,
			}
			origDestConnectionID = c.OriginalDestConnectionID
		}
	}
	if !s.config.AcceptToken(p.remoteAddr, token) {
		go func() {
			defer p.buffer.Release()
			if token != nil && token.IsRetryToken {
				if err := s.maybeSendInvalidToken(p, hdr); err != nil {
					s.logger.Debugf("Error sending INVALID_TOKEN error: %s", err)
				}
				return
			}
			if err := s.sendRetry(p.remoteAddr, hdr); err != nil {
				s.logger.Debugf("Error sending Retry: %s", err)
			}
		}()
		return nil
	}

	if queueLen := atomic.LoadInt32(&s.sessionQueueLen); queueLen >= protocol.MaxAcceptQueueSize {
		s.logger.Debugf("Rejecting new connection. Server currently busy. Accept queue length: %d (max %d)", queueLen, protocol.MaxAcceptQueueSize)
		go func() {
			if err := s.sendServerBusy(p.remoteAddr, hdr); err != nil {
				s.logger.Debugf("Error rejecting connection: %s", err)
			}
		}()
		return nil
	}

	connID, err := protocol.GenerateConnectionID(s.config.ConnectionIDLength)
	if err != nil {
		return err
	}
	s.logger.Debugf("Changing connection ID to %s.", connID)
	sess := s.createNewSession(
		p.remoteAddr,
		origDestConnectionID,
		hdr.DestConnectionID,
		hdr.SrcConnectionID,
		connID,
		hdr.Version,
	)
	if sess == nil {
		p.buffer.Release()
		return nil
	}
	sess.handlePacket(p)
	for {
		p := s.zeroRTTQueue.Dequeue(hdr.DestConnectionID)
		if p == nil {
			break
		}
		sess.handlePacket(p)
	}
	return nil
}

func (s *baseServer) createNewSession(
	remoteAddr net.Addr,
	origDestConnID protocol.ConnectionID,
	clientDestConnID protocol.ConnectionID,
	destConnID protocol.ConnectionID,
	srcConnID protocol.ConnectionID,
	version protocol.VersionNumber,
) quicSession {
	var qlogger qlog.Tracer
	if s.config.GetLogWriter != nil {
		// Use the same connection ID that is passed to the client's GetLogWriter callback.
		connID := clientDestConnID
		if origDestConnID.Len() > 0 {
			connID = origDestConnID
		}
		if w := s.config.GetLogWriter(connID); w != nil {
			qlogger = qlog.NewTracer(w, protocol.PerspectiveServer, connID)
		}
	}
	sess := s.newSession(
		&conn{pconn: s.conn, currentAddr: remoteAddr},
		s.sessionHandler,
		origDestConnID,
		clientDestConnID,
		destConnID,
		srcConnID,
		s.sessionHandler.GetStatelessResetToken(srcConnID),
		s.config,
		s.tlsConf,
		s.tokenGenerator,
		s.acceptEarlySessions,
		qlogger,
		s.logger,
		version,
	)
	if added := s.sessionHandler.Add(clientDestConnID, sess); !added {
		// We're already keeping track of this connection ID.
		// This might happen if we receive two copies of the Initial at the same time.
		return nil
	}
	s.sessionHandler.Add(srcConnID, sess)
	go sess.run()
	go s.handleNewSession(sess)
	return sess
}

func (s *baseServer) handleNewSession(sess quicSession) {
	sessCtx := sess.Context()
	if s.acceptEarlySessions {
		// wait until the early session is ready (or the handshake fails)
		select {
		case <-sess.earlySessionReady():
		case <-sessCtx.Done():
			return
		}
	} else {
		// wait until the handshake is complete (or fails)
		select {
		case <-sess.HandshakeComplete().Done():
		case <-sessCtx.Done():
			return
		}
	}

	atomic.AddInt32(&s.sessionQueueLen, 1)
	select {
	case s.sessionQueue <- sess:
		// blocks until the session is accepted
	case <-sessCtx.Done():
		atomic.AddInt32(&s.sessionQueueLen, -1)
		// don't pass sessions that were already closed to Accept()
	}
}

func (s *baseServer) sendRetry(remoteAddr net.Addr, hdr *wire.Header) error {
	// Log the Initial packet now.
	// If no Retry is sent, the packet will be logged by the session.
	(&wire.ExtendedHeader{Header: *hdr}).Log(s.logger)
	token, err := s.tokenGenerator.NewRetryToken(remoteAddr, hdr.DestConnectionID)
	if err != nil {
		return err
	}
	connID, err := protocol.GenerateConnectionID(s.config.ConnectionIDLength)
	if err != nil {
		return err
	}
	replyHdr := &wire.ExtendedHeader{}
	replyHdr.IsLongHeader = true
	replyHdr.Type = protocol.PacketTypeRetry
	replyHdr.Version = hdr.Version
	replyHdr.SrcConnectionID = connID
	replyHdr.DestConnectionID = hdr.SrcConnectionID
	replyHdr.Token = token
	s.logger.Debugf("Changing connection ID to %s.", connID)
	s.logger.Debugf("-> Sending Retry")
	replyHdr.Log(s.logger)
	buf := &bytes.Buffer{}
	if err := replyHdr.Write(buf, hdr.Version); err != nil {
		return err
	}
	// append the Retry integrity tag
	tag := handshake.GetRetryIntegrityTag(buf.Bytes(), hdr.DestConnectionID)
	buf.Write(tag[:])
	_, err = s.conn.WriteTo(buf.Bytes(), remoteAddr)
	return err
}

func (s *baseServer) maybeSendInvalidToken(p *receivedPacket, hdr *wire.Header) error {
	// Only send INVALID_TOKEN if we can unprotect the packet.
	// This makes sure that we won't send it for packets that were corrupted.
	sealer, opener := handshake.NewInitialAEAD(hdr.DestConnectionID, protocol.PerspectiveServer)
	data := p.data[:hdr.ParsedLen()+hdr.Length]
	extHdr, err := unpackHeader(opener, hdr, data, hdr.Version)
	if err != nil {
		// don't return the error here. Just drop the packet.
		return nil
	}
	hdrLen := extHdr.ParsedLen()
	if _, err := opener.Open(data[hdrLen:hdrLen], data[hdrLen:], extHdr.PacketNumber, data[:hdrLen]); err != nil {
		// don't return the error here. Just drop the packet.
		return nil
	}
	if s.logger.Debug() {
		s.logger.Debugf("Client sent an invalid retry token. Sending INVALID_TOKEN to %s.", p.remoteAddr)
	}
	return s.sendError(p.remoteAddr, hdr, sealer, qerr.InvalidToken)
}

func (s *baseServer) sendServerBusy(remoteAddr net.Addr, hdr *wire.Header) error {
	sealer, _ := handshake.NewInitialAEAD(hdr.DestConnectionID, protocol.PerspectiveServer)
	return s.sendError(remoteAddr, hdr, sealer, qerr.ServerBusy)
}

// sendError sends the error as a response to the packet received with header hdr
func (s *baseServer) sendError(remoteAddr net.Addr, hdr *wire.Header, sealer handshake.LongHeaderSealer, errorCode qerr.ErrorCode) error {
	packetBuffer := getPacketBuffer()
	defer packetBuffer.Release()
	buf := bytes.NewBuffer(packetBuffer.Data)

	ccf := &wire.ConnectionCloseFrame{ErrorCode: errorCode}

	replyHdr := &wire.ExtendedHeader{}
	replyHdr.IsLongHeader = true
	replyHdr.Type = protocol.PacketTypeInitial
	replyHdr.Version = hdr.Version
	replyHdr.SrcConnectionID = hdr.DestConnectionID
	replyHdr.DestConnectionID = hdr.SrcConnectionID
	replyHdr.PacketNumberLen = protocol.PacketNumberLen4
	replyHdr.Length = 4 /* packet number len */ + ccf.Length(hdr.Version) + protocol.ByteCount(sealer.Overhead())
	if err := replyHdr.Write(buf, hdr.Version); err != nil {
		return err
	}
	payloadOffset := buf.Len()

	if err := ccf.Write(buf, hdr.Version); err != nil {
		return err
	}

	raw := buf.Bytes()
	_ = sealer.Seal(raw[payloadOffset:payloadOffset], raw[payloadOffset:], replyHdr.PacketNumber, raw[:payloadOffset])
	raw = raw[0 : buf.Len()+sealer.Overhead()]

	pnOffset := payloadOffset - int(replyHdr.PacketNumberLen)
	sealer.EncryptHeader(
		raw[pnOffset+4:pnOffset+4+16],
		&raw[0],
		raw[pnOffset:payloadOffset],
	)

	replyHdr.Log(s.logger)
	wire.LogFrame(s.logger, ccf, true)
	_, err := s.conn.WriteTo(raw, remoteAddr)
	return err
}

func (s *baseServer) sendVersionNegotiationPacket(p *receivedPacket, hdr *wire.Header) {
	s.logger.Debugf("Client offered version %s, sending Version Negotiation", hdr.Version)
	data, err := wire.ComposeVersionNegotiation(hdr.SrcConnectionID, hdr.DestConnectionID, s.config.Versions)
	if err != nil {
		s.logger.Debugf("Error composing Version Negotiation: %s", err)
		return
	}
	if _, err := s.conn.WriteTo(data, p.remoteAddr); err != nil {
		s.logger.Debugf("Error sending Version Negotiation: %s", err)
	}
}
