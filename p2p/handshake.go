package p2p

import (
	"crypto/ecdsa"
	"crypto/rand"
	"errors"
	"fmt"
	"io"
	"net"

	"github.com/ethereum/go-ethereum/crypto"
	"github.com/ethereum/go-ethereum/crypto/ecies"
	"github.com/ethereum/go-ethereum/crypto/secp256k1"
	"github.com/ethereum/go-ethereum/p2p/discover"
	"github.com/ethereum/go-ethereum/rlp"
)

const (
	sskLen = 16 // ecies.MaxSharedKeyLength(pubKey) / 2
	sigLen = 65 // elliptic S256
	pubLen = 64 // 512 bit pubkey in uncompressed representation without format byte
	shaLen = 32 // hash length (for nonce etc)

	authMsgLen  = sigLen + shaLen + pubLen + shaLen + 1
	authRespLen = pubLen + shaLen + 1

	eciesBytes = 65 + 16 + 32
	iHSLen     = authMsgLen + eciesBytes  // size of the final ECIES payload sent as initiator's handshake
	rHSLen     = authRespLen + eciesBytes // size of the final ECIES payload sent as receiver's handshake
)

type conn struct {
	*frameRW
	*protoHandshake
}

func newConn(fd net.Conn, hs *protoHandshake) *conn {
	return &conn{newFrameRW(fd, msgWriteTimeout), hs}
}

// encHandshake represents information about the remote end
// of a connection that is negotiated during the encryption handshake.
type encHandshake struct {
	ID         discover.NodeID
	IngressMAC []byte
	EgressMAC  []byte
	Token      []byte
}

// protoHandshake is the RLP structure of the protocol handshake.
type protoHandshake struct {
	Version    uint64
	Name       string
	Caps       []Cap
	ListenPort uint64
	ID         discover.NodeID
}

// setupConn starts a protocol session on the given connection.
// It runs the encryption handshake and the protocol handshake.
// If dial is non-nil, the connection the local node is the initiator.
func setupConn(fd net.Conn, prv *ecdsa.PrivateKey, our *protoHandshake, dial *discover.Node) (*conn, error) {
	if dial == nil {
		return setupInboundConn(fd, prv, our)
	} else {
		return setupOutboundConn(fd, prv, our, dial)
	}
}

func setupInboundConn(fd net.Conn, prv *ecdsa.PrivateKey, our *protoHandshake) (*conn, error) {
	// var remotePubkey []byte
	// sessionToken, remotePubkey, err = inboundEncHandshake(fd, prv, nil)
	// copy(remoteID[:], remotePubkey)

	rw := newFrameRW(fd, msgWriteTimeout)
	rhs, err := readProtocolHandshake(rw, our)
	if err != nil {
		return nil, err
	}
	if err := writeProtocolHandshake(rw, our); err != nil {
		return nil, fmt.Errorf("protocol write error: %v", err)
	}
	return &conn{rw, rhs}, nil
}

func setupOutboundConn(fd net.Conn, prv *ecdsa.PrivateKey, our *protoHandshake, dial *discover.Node) (*conn, error) {
	// remoteID = dial.ID
	// sessionToken, err = outboundEncHandshake(fd, prv, remoteID[:], nil)

	rw := newFrameRW(fd, msgWriteTimeout)
	if err := writeProtocolHandshake(rw, our); err != nil {
		return nil, fmt.Errorf("protocol write error: %v", err)
	}
	rhs, err := readProtocolHandshake(rw, our)
	if err != nil {
		return nil, fmt.Errorf("protocol handshake read error: %v", err)
	}
	if rhs.ID != dial.ID {
		return nil, errors.New("dialed node id mismatch")
	}
	return &conn{rw, rhs}, nil
}

// outboundEncHandshake negotiates a session token on conn.
// it should be called on the dialing side of the connection.
//
// privateKey is the local client's private key
// remotePublicKey is the remote peer's node ID
// sessionToken is the token from a previous session with this node.
func outboundEncHandshake(conn io.ReadWriter, prvKey *ecdsa.PrivateKey, remotePublicKey []byte, sessionToken []byte) (
	newSessionToken []byte,
	err error,
) {
	auth, initNonce, randomPrivKey, err := authMsg(prvKey, remotePublicKey, sessionToken)
	if err != nil {
		return nil, err
	}
	if _, err = conn.Write(auth); err != nil {
		return nil, err
	}

	response := make([]byte, rHSLen)
	if _, err = io.ReadFull(conn, response); err != nil {
		return nil, err
	}
	recNonce, remoteRandomPubKey, _, err := completeHandshake(response, prvKey)
	if err != nil {
		return nil, err
	}

	return newSession(initNonce, recNonce, randomPrivKey, remoteRandomPubKey)
}

// authMsg creates the initiator handshake.
func authMsg(prvKey *ecdsa.PrivateKey, remotePubKeyS, sessionToken []byte) (
	auth, initNonce []byte,
	randomPrvKey *ecdsa.PrivateKey,
	err error,
) {
	// session init, common to both parties
	remotePubKey, err := importPublicKey(remotePubKeyS)
	if err != nil {
		return
	}

	var tokenFlag byte // = 0x00
	if sessionToken == nil {
		// no session token found means we need to generate shared secret.
		// ecies shared secret is used as initial session token for new peers
		// generate shared key from prv and remote pubkey
		if sessionToken, err = ecies.ImportECDSA(prvKey).GenerateShared(ecies.ImportECDSAPublic(remotePubKey), sskLen, sskLen); err != nil {
			return
		}
		// tokenFlag = 0x00 // redundant
	} else {
		// for known peers, we use stored token from the previous session
		tokenFlag = 0x01
	}

	//E(remote-pubk, S(ecdhe-random, ecdh-shared-secret^nonce) || H(ecdhe-random-pubk) || pubk || nonce || 0x0)
	// E(remote-pubk, S(ecdhe-random, token^nonce) || H(ecdhe-random-pubk) || pubk || nonce || 0x1)
	// allocate msgLen long message,
	var msg []byte = make([]byte, authMsgLen)
	initNonce = msg[authMsgLen-shaLen-1 : authMsgLen-1]
	if _, err = rand.Read(initNonce); err != nil {
		return
	}
	// create known message
	// ecdh-shared-secret^nonce for new peers
	// token^nonce for old peers
	var sharedSecret = xor(sessionToken, initNonce)

	// generate random keypair to use for signing
	if randomPrvKey, err = crypto.GenerateKey(); err != nil {
		return
	}
	// sign shared secret (message known to both parties): shared-secret
	var signature []byte
	// signature = sign(ecdhe-random, shared-secret)
	// uses secp256k1.Sign
	if signature, err = crypto.Sign(sharedSecret, randomPrvKey); err != nil {
		return
	}

	// message
	// signed-shared-secret || H(ecdhe-random-pubk) || pubk || nonce || 0x0
	copy(msg, signature) // copy signed-shared-secret
	// H(ecdhe-random-pubk)
	var randomPubKey64 []byte
	if randomPubKey64, err = exportPublicKey(&randomPrvKey.PublicKey); err != nil {
		return
	}
	var pubKey64 []byte
	if pubKey64, err = exportPublicKey(&prvKey.PublicKey); err != nil {
		return
	}
	copy(msg[sigLen:sigLen+shaLen], crypto.Sha3(randomPubKey64))
	// pubkey copied to the correct segment.
	copy(msg[sigLen+shaLen:sigLen+shaLen+pubLen], pubKey64)
	// nonce is already in the slice
	// stick tokenFlag byte to the end
	msg[authMsgLen-1] = tokenFlag

	// encrypt using remote-pubk
	// auth = eciesEncrypt(remote-pubk, msg)
	if auth, err = crypto.Encrypt(remotePubKey, msg); err != nil {
		return
	}
	return
}

// completeHandshake is called when the initiator receives an
// authentication response (aka receiver handshake). It completes the
// handshake by reading off parameters the remote peer provides needed
// to set up the secure session.
func completeHandshake(auth []byte, prvKey *ecdsa.PrivateKey) (
	respNonce []byte,
	remoteRandomPubKey *ecdsa.PublicKey,
	tokenFlag bool,
	err error,
) {
	var msg []byte
	// they prove that msg is meant for me,
	// I prove I possess private key if i can read it
	if msg, err = crypto.Decrypt(prvKey, auth); err != nil {
		return
	}

	respNonce = msg[pubLen : pubLen+shaLen]
	var remoteRandomPubKeyS = msg[:pubLen]
	if remoteRandomPubKey, err = importPublicKey(remoteRandomPubKeyS); err != nil {
		return
	}
	if msg[authRespLen-1] == 0x01 {
		tokenFlag = true
	}
	return
}

// inboundEncHandshake negotiates a session token on conn.
// it should be called on the listening side of the connection.
//
// privateKey is the local client's private key
// sessionToken is the token from a previous session with this node.
func inboundEncHandshake(conn io.ReadWriter, prvKey *ecdsa.PrivateKey, sessionToken []byte) (
	token, remotePubKey []byte,
	err error,
) {
	// we are listening connection. we are responders in the
	// handshake. Extract info from the authentication. The initiator
	// starts by sending us a handshake that we need to respond to. so
	// we read auth message first, then respond.
	auth := make([]byte, iHSLen)
	if _, err := io.ReadFull(conn, auth); err != nil {
		return nil, nil, err
	}
	response, recNonce, initNonce, remotePubKey, randomPrivKey, remoteRandomPubKey, err := authResp(auth, sessionToken, prvKey)
	if err != nil {
		return nil, nil, err
	}
	if _, err = conn.Write(response); err != nil {
		return nil, nil, err
	}
	token, err = newSession(initNonce, recNonce, randomPrivKey, remoteRandomPubKey)
	return token, remotePubKey, err
}

// authResp is called by peer if it accepted (but not
// initiated) the connection from the remote. It is passed the initiator
// handshake received and the session token belonging to the
// remote initiator.
//
// The first return value is the authentication response (aka receiver
// handshake) that is to be sent to the remote initiator.
func authResp(auth, sessionToken []byte, prvKey *ecdsa.PrivateKey) (
	authResp, respNonce, initNonce, remotePubKeyS []byte,
	randomPrivKey *ecdsa.PrivateKey,
	remoteRandomPubKey *ecdsa.PublicKey,
	err error,
) {
	// they prove that msg is meant for me,
	// I prove I possess private key if i can read it
	msg, err := crypto.Decrypt(prvKey, auth)
	if err != nil {
		return
	}

	remotePubKeyS = msg[sigLen+shaLen : sigLen+shaLen+pubLen]
	remotePubKey, _ := importPublicKey(remotePubKeyS)

	var tokenFlag byte
	if sessionToken == nil {
		// no session token found means we need to generate shared secret.
		// ecies shared secret is used as initial session token for new peers
		// generate shared key from prv and remote pubkey
		if sessionToken, err = ecies.ImportECDSA(prvKey).GenerateShared(ecies.ImportECDSAPublic(remotePubKey), sskLen, sskLen); err != nil {
			return
		}
		// tokenFlag = 0x00 // redundant
	} else {
		// for known peers, we use stored token from the previous session
		tokenFlag = 0x01
	}

	// the initiator nonce is read off the end of the message
	initNonce = msg[authMsgLen-shaLen-1 : authMsgLen-1]
	// I prove that i own prv key (to derive shared secret, and read
	// nonce off encrypted msg) and that I own shared secret they
	// prove they own the private key belonging to ecdhe-random-pubk
	// we can now reconstruct the signed message and recover the peers
	// pubkey
	var signedMsg = xor(sessionToken, initNonce)
	var remoteRandomPubKeyS []byte
	if remoteRandomPubKeyS, err = secp256k1.RecoverPubkey(signedMsg, msg[:sigLen]); err != nil {
		return
	}
	// convert to ECDSA standard
	if remoteRandomPubKey, err = importPublicKey(remoteRandomPubKeyS); err != nil {
		return
	}

	// now we find ourselves a long task too, fill it random
	var resp = make([]byte, authRespLen)
	// generate shaLen long nonce
	respNonce = resp[pubLen : pubLen+shaLen]
	if _, err = rand.Read(respNonce); err != nil {
		return
	}
	// generate random keypair for session
	if randomPrivKey, err = crypto.GenerateKey(); err != nil {
		return
	}
	// responder auth message
	// E(remote-pubk, ecdhe-random-pubk || nonce || 0x0)
	var randomPubKeyS []byte
	if randomPubKeyS, err = exportPublicKey(&randomPrivKey.PublicKey); err != nil {
		return
	}
	copy(resp[:pubLen], randomPubKeyS)
	// nonce is already in the slice
	resp[authRespLen-1] = tokenFlag

	// encrypt using remote-pubk
	// auth = eciesEncrypt(remote-pubk, msg)
	// why not encrypt with ecdhe-random-remote
	if authResp, err = crypto.Encrypt(remotePubKey, resp); err != nil {
		return
	}
	return
}

// newSession is called after the handshake is completed. The
// arguments are values negotiated in the handshake. The return value
// is a new session Token to be remembered for the next time we
// connect with this peer.
func newSession(initNonce, respNonce []byte, privKey *ecdsa.PrivateKey, remoteRandomPubKey *ecdsa.PublicKey) ([]byte, error) {
	// 3) Now we can trust ecdhe-random-pubk to derive new keys
	//ecdhe-shared-secret = ecdh.agree(ecdhe-random, remote-ecdhe-random-pubk)
	pubKey := ecies.ImportECDSAPublic(remoteRandomPubKey)
	dhSharedSecret, err := ecies.ImportECDSA(privKey).GenerateShared(pubKey, sskLen, sskLen)
	if err != nil {
		return nil, err
	}
	sharedSecret := crypto.Sha3(dhSharedSecret, crypto.Sha3(respNonce, initNonce))
	sessionToken := crypto.Sha3(sharedSecret)
	return sessionToken, nil
}

// importPublicKey unmarshals 512 bit public keys.
func importPublicKey(pubKey []byte) (pubKeyEC *ecdsa.PublicKey, err error) {
	var pubKey65 []byte
	switch len(pubKey) {
	case 64:
		// add 'uncompressed key' flag
		pubKey65 = append([]byte{0x04}, pubKey...)
	case 65:
		pubKey65 = pubKey
	default:
		return nil, fmt.Errorf("invalid public key length %v (expect 64/65)", len(pubKey))
	}
	return crypto.ToECDSAPub(pubKey65), nil
}

func exportPublicKey(pubKeyEC *ecdsa.PublicKey) (pubKey []byte, err error) {
	if pubKeyEC == nil {
		return nil, fmt.Errorf("no ECDSA public key given")
	}
	return crypto.FromECDSAPub(pubKeyEC)[1:], nil
}

func xor(one, other []byte) (xor []byte) {
	xor = make([]byte, len(one))
	for i := 0; i < len(one); i++ {
		xor[i] = one[i] ^ other[i]
	}
	return xor
}

func writeProtocolHandshake(w MsgWriter, our *protoHandshake) error {
	return EncodeMsg(w, handshakeMsg, our.Version, our.Name, our.Caps, our.ListenPort, our.ID[:])
}

func readProtocolHandshake(r MsgReader, our *protoHandshake) (*protoHandshake, error) {
	// read and handle remote handshake
	msg, err := r.ReadMsg()
	if err != nil {
		return nil, err
	}
	if msg.Code == discMsg {
		// disconnect before protocol handshake is valid according to the
		// spec and we send it ourself if Server.addPeer fails.
		var reason DiscReason
		rlp.Decode(msg.Payload, &reason)
		return nil, discRequestedError(reason)
	}
	if msg.Code != handshakeMsg {
		return nil, fmt.Errorf("expected handshake, got %x", msg.Code)
	}
	if msg.Size > baseProtocolMaxMsgSize {
		return nil, fmt.Errorf("message too big (%d > %d)", msg.Size, baseProtocolMaxMsgSize)
	}
	var hs protoHandshake
	if err := msg.Decode(&hs); err != nil {
		return nil, err
	}
	// validate handshake info
	if hs.Version != our.Version {
		return nil, newPeerError(errP2PVersionMismatch, "required version %d, received %d\n", baseProtocolVersion, hs.Version)
	}
	if (hs.ID == discover.NodeID{}) {
		return nil, newPeerError(errPubkeyInvalid, "missing")
	}
	return &hs, nil
}
