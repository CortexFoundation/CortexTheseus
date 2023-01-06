package filecache

import "time"

type policyItemKey interface {
	Before(policyItemKey) bool
}

type Policy interface {
	Choose() policyItemKey
	Used(k policyItemKey, at time.Time)
	Forget(k policyItemKey)
	NumItems() int
}
