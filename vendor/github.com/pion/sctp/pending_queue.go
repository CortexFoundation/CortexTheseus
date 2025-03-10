// SPDX-FileCopyrightText: 2023 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package sctp

import (
	"errors"
)

// pendingBaseQueue

type pendingBaseQueue struct {
	queue []*chunkPayloadData
}

func newPendingBaseQueue() *pendingBaseQueue {
	return &pendingBaseQueue{queue: []*chunkPayloadData{}}
}

func (q *pendingBaseQueue) push(c *chunkPayloadData) {
	q.queue = append(q.queue, c)
}

func (q *pendingBaseQueue) pop() *chunkPayloadData {
	if len(q.queue) == 0 {
		return nil
	}
	c := q.queue[0]
	q.queue = q.queue[1:]

	return c
}

func (q *pendingBaseQueue) get(i int) *chunkPayloadData {
	if len(q.queue) == 0 || i < 0 || i >= len(q.queue) {
		return nil
	}

	return q.queue[i]
}

func (q *pendingBaseQueue) size() int {
	return len(q.queue)
}

// pendingQueue

type pendingQueue struct {
	unorderedQueue      *pendingBaseQueue
	orderedQueue        *pendingBaseQueue
	nBytes              int
	selected            bool
	unorderedIsSelected bool
}

// Pending queue errors.
var (
	ErrUnexpectedChuckPoppedUnordered = errors.New("unexpected chunk popped (unordered)")
	ErrUnexpectedChuckPoppedOrdered   = errors.New("unexpected chunk popped (ordered)")
	ErrUnexpectedQState               = errors.New("unexpected q state (should've been selected)")
)

func newPendingQueue() *pendingQueue {
	return &pendingQueue{
		unorderedQueue: newPendingBaseQueue(),
		orderedQueue:   newPendingBaseQueue(),
	}
}

func (q *pendingQueue) push(c *chunkPayloadData) {
	if c.unordered {
		q.unorderedQueue.push(c)
	} else {
		q.orderedQueue.push(c)
	}
	q.nBytes += len(c.userData)
}

func (q *pendingQueue) peek() *chunkPayloadData {
	if q.selected {
		if q.unorderedIsSelected {
			return q.unorderedQueue.get(0)
		}

		return q.orderedQueue.get(0)
	}

	if c := q.unorderedQueue.get(0); c != nil {
		return c
	}

	return q.orderedQueue.get(0)
}

func (q *pendingQueue) pop(chunkPayload *chunkPayloadData) error { //nolint:cyclop
	if q.selected { //nolint:nestif
		var popped *chunkPayloadData
		if q.unorderedIsSelected {
			popped = q.unorderedQueue.pop()
			if popped != chunkPayload {
				return ErrUnexpectedChuckPoppedUnordered
			}
		} else {
			popped = q.orderedQueue.pop()
			if popped != chunkPayload {
				return ErrUnexpectedChuckPoppedOrdered
			}
		}
		if popped.endingFragment {
			q.selected = false
		}
	} else {
		if !chunkPayload.beginningFragment {
			return ErrUnexpectedQState
		}
		if chunkPayload.unordered {
			popped := q.unorderedQueue.pop()
			if popped != chunkPayload {
				return ErrUnexpectedChuckPoppedUnordered
			}
			if !popped.endingFragment {
				q.selected = true
				q.unorderedIsSelected = true
			}
		} else {
			popped := q.orderedQueue.pop()
			if popped != chunkPayload {
				return ErrUnexpectedChuckPoppedOrdered
			}
			if !popped.endingFragment {
				q.selected = true
				q.unorderedIsSelected = false
			}
		}
	}
	q.nBytes -= len(chunkPayload.userData)

	return nil
}

func (q *pendingQueue) getNumBytes() int {
	return q.nBytes
}

func (q *pendingQueue) size() int {
	return q.unorderedQueue.size() + q.orderedQueue.size()
}
