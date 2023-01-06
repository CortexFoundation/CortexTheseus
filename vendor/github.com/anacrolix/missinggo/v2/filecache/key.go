package filecache

type key string

func (me key) Before(other policyItemKey) bool {
	return me < other.(key)
}
