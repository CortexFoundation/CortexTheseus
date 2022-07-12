// Copyright 2018 The CortexTheseus Authors
// This file is part of the CortexFoundation library.
//
// The CortexFoundation library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexFoundation library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with the CortexFoundation library. If not, see <http://www.gnu.org/licenses/>.

package sha3

import (
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/CortexFoundation/CortexTheseus/log"

	"github.com/CortexFoundation/CortexTheseus/common"
	"github.com/CortexFoundation/CortexTheseus/common/hexutil"
	"github.com/CortexFoundation/CortexTheseus/core/types"
)

var errSHAThreeStopped = errors.New("sha3 stopped")

// API exposes sha3 related methods for the RPC interface.
type API struct {
	sha3 *SHAThree // Make sure the mode of sha3 is normal.
}

// GetWork returns a work package for external miner.
//
// The work package consists of 3 strings:
//   result[0] - 32 bytes hex encoded current block header pow-hash
//   result[1] - 32 bytes hex encoded seed hash used for DAG
//   result[2] - 32 bytes hex encoded boundary condition ("target"), 2^256/difficulty
func (api *API) GetWork() ([4]string, error) {
	if api.sha3.config.PowMode != ModeNormal && api.sha3.config.PowMode != ModeTest {
		return [4]string{}, errors.New("not supported")
	}

	var (
		workCh = make(chan [4]string, 1)
		errc   = make(chan error, 1)
	)

	select {
	case api.sha3.fetchWorkCh <- &sealWork{errc: errc, res: workCh}:
	case <-api.sha3.exitCh:
		return [4]string{}, errSHAThreeStopped
	}

	select {
	case work := <-workCh:
		return work, nil
	case err := <-errc:
		return [4]string{}, err
	}
}

// SubmitWork can be used by external miner to submit their POW solution.
// It returns an indication if the work was accepted.
// Note either an invalid solution, a stale work a non-existent work will return false.
func (api *API) SubmitWork(nonce types.BlockNonce, hash common.Hash, solution string) bool {
	var sol types.BlockSolution
	solBytes, solErr := hex.DecodeString(solution[2:])
	if solErr != nil {
		log.Warn(fmt.Sprintf("Convert Error %v: ", solErr))
		return false
	}
	sol.UnmarshalText(solBytes)
	if api.sha3.config.PowMode != ModeNormal && api.sha3.config.PowMode != ModeTest {
		return false
	}

	var errc = make(chan error, 1)
	select {
	case api.sha3.submitWorkCh <- &mineResult{
		nonce: nonce,
		//mixDigest: digest,
		hash:     hash,
		errc:     errc,
		solution: sol,
	}:
	case <-api.sha3.exitCh:
		return false
	}

	err := <-errc
	return err == nil
}

// SubmitHashrate can be used for remote miners to submit their hash rate.
// This enables the node to report the combined hash rate of all miners
// which submit work through this node.
//
// It accepts the miner hash rate and an identifier which must be unique
// between nodes.
func (api *API) SubmitHashRate(rate hexutil.Uint64, id common.Hash) bool {
	if api.sha3.config.PowMode != ModeNormal && api.sha3.config.PowMode != ModeTest {
		return false
	}

	var done = make(chan struct{}, 1)

	select {
	case api.sha3.submitRateCh <- &hashrate{done: done, rate: uint64(rate), id: id}:
	case <-api.sha3.exitCh:
		return false
	}

	// Block until hash rate submitted successfully.
	<-done

	return true
}

// GetHashrate returns the current hashrate for local CPU miner and remote miner.
func (api *API) GetHashrate() uint64 {
	return uint64(api.sha3.Hashrate())
}
