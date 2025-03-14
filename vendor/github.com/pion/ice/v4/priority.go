// SPDX-FileCopyrightText: 2023 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

package ice

import (
	"encoding/binary"

	"github.com/pion/stun/v3"
)

// PriorityAttr represents PRIORITY attribute.
type PriorityAttr uint32

const prioritySize = 4 // 32 bit

// AddTo adds PRIORITY attribute to message.
func (p PriorityAttr) AddTo(m *stun.Message) error {
	v := make([]byte, prioritySize)
	binary.BigEndian.PutUint32(v, uint32(p))
	m.Add(stun.AttrPriority, v)

	return nil
}

// GetFrom decodes PRIORITY attribute from message.
func (p *PriorityAttr) GetFrom(m *stun.Message) error {
	v, err := m.Get(stun.AttrPriority)
	if err != nil {
		return err
	}
	if err = stun.CheckSize(stun.AttrPriority, len(v), prioritySize); err != nil {
		return err
	}
	*p = PriorityAttr(binary.BigEndian.Uint32(v))

	return nil
}
