package merkletree

type BlockContent struct {
	x    string
	n    uint64
	hash []byte
}

func (t BlockContent) cache() []byte {
	return t.hash
}

func (t BlockContent) CalculateHash() ([]byte, error) {
	if cache := t.cache(); cache != nil {
		return cache, nil
	}

	h := newHasher()
	defer returnHasherToPool(h)
	t.hash = h.sum([]byte(t.x))
	return t.hash, nil
}

func (t BlockContent) Equals(other Content) (bool, error) {
	return t.x == other.(BlockContent).x && t.n == other.(BlockContent).n, nil
}

func NewContent(x string, n uint64) BlockContent {
	return BlockContent{x: x, n: n}
}

func (t BlockContent) N() uint64 {
	return t.n
}
