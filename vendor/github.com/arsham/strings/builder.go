package strings

import (
	"io"
	"unicode/utf8"
	"unsafe"
)

// A Builder is used to efficiently build a string using Write methods. It
// minimizes memory copying. The zero value is ready to use. Do not copy a non-
// zero Builder.
type Builder struct {
	addr     *Builder // of receiver, to detect copies by value
	buf      []byte
	off      int    // read at &buf[off], write at &buf[len(buf)]
	lastRead readOp // last read operation, so that Unread* can work correctly.
}

// The readOp constants describe the last action performed on the buffer, so
// that UnreadRune and UnreadByte can check for invalid usage. opReadRuneX
// constants are chosen such that converted to int they correspond to the rune
// size that was read.
type readOp int8

// Don't use iota for these, as the values need to correspond with the names and
// comments, which is easier to see when being explicit.
const (
	opRead    readOp = -1 // Any other read operation.
	opInvalid readOp = 0  // Non-read operation.
)

// noescape hides a pointer from escape analysis. noescape is the identity
// function but escape analysis doesn't think the output depends on the input.
// noescape is inlined and currently compiles down to zero instructions. USE
// CAREFULLY! This was copied from the runtime; see issues 23382 and 7921.
//go:nosplit
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

func (b *Builder) copyCheck() {
	if b.addr == nil {
		// This hack works around a failing of Go's escape analysis that was
		// causing b to escape and be heap allocated.
		// See issue 23382.
		// TODO: once issue 7921 is fixed, this should be reverted to just
		// "b.addr = b".
		b.addr = (*Builder)(noescape(unsafe.Pointer(b)))
	} else if b.addr != b {
		panic("strings: illegal use of non-zero Builder copied by value")
	}
}

// String returns the accumulated string. It doesn't matter if the Read() method
// has been called, it always returns the contents.
func (b *Builder) String() string {
	return *(*string)(unsafe.Pointer(&b.buf))
}

// Len returns the number of accumulated bytes; b.Len() == len(b.String()).
func (b *Builder) Len() int { return len(b.buf) }

// Reset resets the Builder to be empty.
func (b *Builder) Reset() {
	b.addr = nil
	b.buf = nil
}

// grow copies the buffer to a new, larger buffer so that there are at least n
// bytes of capacity beyond len(b.buf).
func (b *Builder) grow(n int) {
	buf := make([]byte, len(b.buf), 2*cap(b.buf)+n)
	copy(buf, b.buf)
	b.buf = buf
}

// Grow grows b's capacity, if necessary, to guarantee space for another n
// bytes. After Grow(n), at least n bytes can be written to b without another
// allocation. If n is negative, Grow panics.
func (b *Builder) Grow(n int) {
	b.copyCheck()
	if n < 0 {
		panic("strings.Builder.Grow: negative count")
	}
	if cap(b.buf)-len(b.buf) < n {
		b.grow(n)
	}
}

// Write appends the contents of p to b's buffer. Write always returns len(p),
// nil.
func (b *Builder) Write(p []byte) (int, error) {
	b.copyCheck()
	b.lastRead = opInvalid
	b.buf = append(b.buf, p...)
	return len(p), nil
}

// WriteByte appends the byte c to b's buffer. The returned error is always nil.
func (b *Builder) WriteByte(c byte) error {
	b.copyCheck()
	b.lastRead = opInvalid
	b.buf = append(b.buf, c)
	return nil
}

// WriteBytes appends the s to b's buffer. The returned error is always nil.
func (b *Builder) WriteBytes(s []byte) (int, error) {
	b.copyCheck()
	b.lastRead = opInvalid
	b.buf = append(b.buf, s...)
	return len(s), nil
}

// WriteRune appends the UTF-8 encoding of Unicode code point r to b's buffer.
// It returns the length of r and a nil error.
func (b *Builder) WriteRune(r rune) (int, error) {
	b.copyCheck()
	b.lastRead = opInvalid
	if r < utf8.RuneSelf {
		b.buf = append(b.buf, byte(r))
		return 1, nil
	}
	l := len(b.buf)
	if cap(b.buf)-l < utf8.UTFMax {
		b.grow(utf8.UTFMax)
	}
	n := utf8.EncodeRune(b.buf[l:l+utf8.UTFMax], r)
	b.buf = b.buf[:l+n]
	return n, nil
}

// WriteString appends the contents of s to b's buffer. It returns the length of
// s and a nil error.
func (b *Builder) WriteString(s string) (int, error) {
	b.copyCheck()
	b.lastRead = opInvalid
	b.buf = append(b.buf, s...)
	return len(s), nil
}

// Bytes returns a slice of length b.Len() holding the unread portion of the
// buffer. The slice is valid for use only until the next buffer modification
// (that is, only until the next call to a method like Read, Write, Reset, or
// Truncate). The slice aliases the buffer content at least until the next
// buffer modification, so immediate changes to the slice will affect the result
// of future reads.
func (b *Builder) Bytes() []byte { return b.buf[b.off:] }

// Read reads the next len(p) bytes from the Builder or until all contents are
// read. The return value n is the number of bytes read. If the Builder has no
// data to return, err is io.EOF (unless len(p) is zero); otherwise it is nil.
// This method doesn't alter the internal buffer.
func (b *Builder) Read(p []byte) (n int, err error) {
	b.lastRead = opInvalid
	if len(b.buf) <= b.off {
		b.lastRead = opInvalid
		if len(p) == 0 {
			return 0, nil
		}
		return 0, io.EOF
	}
	n = copy(p, b.buf[b.off:])
	b.off += n
	if n > 0 {
		b.lastRead = opRead
	}
	return n, nil
}
