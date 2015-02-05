package p2p

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/ethereum/go-ethereum/crypto"
)

func TestCryptoHandshake(t *testing.T) {
	var err error
	var sessionToken []byte
	prv0, _ := crypto.GenerateKey()
	pub0 := &prv0.PublicKey
	prv1, _ := crypto.GenerateKey()
	pub1 := &prv1.PublicKey

	var initiator, receiver *cryptoId
	if initiator, err = newCryptoId(&peerId{crypto.FromECDSA(prv0), crypto.FromECDSAPub(pub0)}); err != nil {
		return
	}
	if receiver, err = newCryptoId(&peerId{crypto.FromECDSA(prv1), crypto.FromECDSAPub(pub1)}); err != nil {
		return
	}

	// simulate handshake by feeding output to input
	// initiator sends handshake 'auth'
	auth, initNonce, randomPrivKey, _, _ := initiator.startHandshake(receiver.pubKeyDER, sessionToken)
	// receiver reads auth and responds with response
	response, remoteRecNonce, remoteInitNonce, remoteRandomPrivKey, _ := receiver.respondToHandshake(auth, crypto.FromECDSAPub(pub0), sessionToken)
	// initiator reads receiver's response and the key exchange completes
	recNonce, remoteRandomPubKey, _, _ := initiator.completeHandshake(response)

	// now both parties should have the same session parameters
	initSessionToken, initSecretRW, _ := initiator.newSession(initNonce, recNonce, auth, randomPrivKey, remoteRandomPubKey)
	recSessionToken, recSecretRW, _ := receiver.newSession(remoteInitNonce, remoteRecNonce, auth, remoteRandomPrivKey, &randomPrivKey.PublicKey)

	fmt.Printf("%x\n%x\n%x\n%x\n%x\n%x\n%x\n%x\n%x\n%x\n", auth, initNonce, response, remoteRecNonce, remoteInitNonce, remoteRandomPubKey, recNonce, &randomPrivKey.PublicKey, initSessionToken, initSecretRW)

	if !bytes.Equal(initNonce, remoteInitNonce) {
		t.Errorf("nonces do not match")
	}
	if !bytes.Equal(recNonce, remoteRecNonce) {
		t.Errorf("receiver nonces do not match")
	}
	if !bytes.Equal(initSessionToken, recSessionToken) {
		t.Errorf("session tokens do not match")
	}
	// aesSecret, macSecret, egressMac, ingressMac
	if !bytes.Equal(initSecretRW.aesSecret, recSecretRW.aesSecret) {
		t.Errorf("AES secrets do not match")
	}
	if !bytes.Equal(initSecretRW.macSecret, recSecretRW.macSecret) {
		t.Errorf("macSecrets do not match")
	}
	if !bytes.Equal(initSecretRW.egressMac, recSecretRW.egressMac) {
		t.Errorf("egressMacs do not match")
	}
	if !bytes.Equal(initSecretRW.ingressMac, recSecretRW.ingressMac) {
		t.Errorf("ingressMacs do not match")
	}

}
