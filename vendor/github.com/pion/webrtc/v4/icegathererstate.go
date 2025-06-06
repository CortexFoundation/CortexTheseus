// SPDX-FileCopyrightText: 2023 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package webrtc

import (
	"sync/atomic"
)

// ICEGathererState represents the current state of the ICE gatherer.
type ICEGathererState uint32

const (
	// ICEGathererStateUnknown is the enum's zero-value.
	ICEGathererStateUnknown ICEGathererState = iota

	// ICEGathererStateNew indicates object has been created but
	// gather() has not been called.
	ICEGathererStateNew

	// ICEGathererStateGathering indicates gather() has been called,
	// and the ICEGatherer is in the process of gathering candidates.
	ICEGathererStateGathering

	// ICEGathererStateComplete indicates the ICEGatherer has completed gathering.
	ICEGathererStateComplete

	// ICEGathererStateClosed indicates the closed state can only be entered
	// when the ICEGatherer has been closed intentionally by calling close().
	ICEGathererStateClosed
)

func (s ICEGathererState) String() string {
	switch s {
	case ICEGathererStateNew:
		return "new"
	case ICEGathererStateGathering:
		return "gathering"
	case ICEGathererStateComplete:
		return "complete"
	case ICEGathererStateClosed:
		return "closed"
	default:
		return ErrUnknownType.Error()
	}
}

func atomicStoreICEGathererState(state *ICEGathererState, newState ICEGathererState) {
	atomic.StoreUint32((*uint32)(state), uint32(newState))
}

func atomicLoadICEGathererState(state *ICEGathererState) ICEGathererState {
	return ICEGathererState(atomic.LoadUint32((*uint32)(state)))
}
