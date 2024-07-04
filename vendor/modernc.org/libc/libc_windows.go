// Copyright 2020 The Libc Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package libc // import "modernc.org/libc"

import (
	"fmt"
	"math"
	mbits "math/bits"
	"os"
	"reflect"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	//UCRT "unicode"
	"unicode/utf16"
	"unsafe"

	"modernc.org/mathutil"
	"modernc.org/memory"
)

const iobEntries = 20

var (
	allocator      memory.Allocator
	allocatorMu    sync.Mutex
	cR             = [...]byte{'r', 0}
	cW             = [...]byte{'w', 0}
	dll            = syscall.NewLazyDLL("ucrtbase.dll")
	iob            [iobEntries]uintptr
	isWindows      = true
	kdll           = syscall.NewLazyDLL("kernel32.dll")
	objects        = map[uintptr]any{}
	objectsMu      sync.Mutex
	threadCallback uintptr
	token          atomic.Uintptr
	uintptrSize    = unsafe.Sizeof(uintptr(0))

	wenvValid  atomic.Bool
	wenviron   uintptr
	winEnviron = []uintptr{0}
)

var X__imp__wenviron = uintptr(unsafe.Pointer(&wenviron))
var X_imp___wenviron = uintptr(unsafe.Pointer(&wenviron))
var Xin6addr_any [16]byte
var Xstderr uintptr
var Xstdin uintptr
var Xstdout uintptr
var Xtimezone long // extern long timezone;

func init() {
	threadCallback = syscall.NewCallback(threadProc)
	iob[0] = X_fdopen(nil, 0, uintptr(unsafe.Pointer(&cR[0])))
	iob[1] = X_fdopen(nil, 1, uintptr(unsafe.Pointer(&cW[0])))
	iob[2] = X_fdopen(nil, 2, uintptr(unsafe.Pointer(&cW[0])))
	Xstdin = iob[0]
	Xstdout = iob[1]
	Xstderr = iob[2]
}

func addObject(o any) uintptr {
	t := token.Add(1)
	objectsMu.Lock()
	objects[t] = o
	objectsMu.Unlock()
	return t
}

func getObject(t uintptr) (r any) {
	objectsMu.Lock()
	r = objects[t]
	objectsMu.Unlock()
	return r
}

func removeObject(t uintptr) {
	objectsMu.Lock()
	delete(objects, t)
	objectsMu.Unlock()
}

//TODO- type MemAuditError struct {
//TODO- 	Caller  string
//TODO- 	Message string
//TODO- }

type long = int32
type ulong = uint32

type threadAdapter struct {
	token      uintptr
	tls        *TLS
	param      uintptr
	threadFunc func(*TLS, uintptr) uint32
}

func (t *threadAdapter) run() uintptr {
	t.tls.token = t.token
	r := t.threadFunc(t.tls, t.param)
	t.tls.endthread(r)
	return uintptr(r)
}

func threadProc(p uintptr) uintptr {
	adp, ok := getObject(p).(*threadAdapter)
	if !ok {
		panic("invalid thread")
	}

	return adp.run()
}

var procCreateThread = kdll.NewProc("CreateThread")
var _ = procCreateThread.Addr()

// libkernel32: __attribute__((dllimport)) HANDLE CreateThread (LPSECURITY_ATTRIBUTES lpThreadAttributes, SIZE_T dwStackSize, LPTHREAD_START_ROUTINE lpStartAddress, LPVOID lpParameter, DWORD dwCreationFlags, LPDWORD lpThreadId);
func CreateThread(tls *TLS, attr uintptr, stackSize Tsize_t, fn uintptr, param uintptr, flags uint32, threadID uintptr) (r uintptr) {
	f := (*struct {
		f func(*TLS, uintptr) uint32
	})(unsafe.Pointer(&struct{ uintptr }{fn})).f
	var adapter = threadAdapter{threadFunc: f, tls: NewTLS(), param: param}
	adapter.token = addObject(&adapter)
	r0, _, _ := syscall.SyscallN(procCreateThread.Addr(), attr, uintptr(stackSize), threadCallback, adapter.token, uintptr(flags), threadID)
	return r0
}

// __attribute__ ((__dllimport__)) uintptr_t __attribute__((__cdecl__)) _beginthread(_beginthread_proc_type _StartAddress,unsigned _StackSize,void *_ArgList);
func X_beginthread(tls *TLS, __StartAddress T_beginthread_proc_type, __StackSize uint32, __ArgList uintptr) (r Tuintptr_t) {
	f := (*struct {
		f func(*TLS, uintptr) uint32
	})(unsafe.Pointer(&struct{ uintptr }{__StartAddress})).f
	var adapter = threadAdapter{threadFunc: f, tls: NewTLS(), param: __ArgList}
	adapter.token = addObject(&adapter)
	r0, _, _ := syscall.SyscallN(procCreateThread.Addr(), 0, uintptr(__StackSize), threadCallback, adapter.token, 0, 0)
	return Tuintptr_t(r0)
}

// __attribute__ ((__dllimport__)) uintptr_t __attribute__((__cdecl__)) _beginthreadex(void *_Security,unsigned _StackSize,_beginthreadex_proc_type _StartAddress,void *_ArgList,unsigned _InitFlag,unsigned *_ThrdAddr);
func X_beginthreadex(tls *TLS, __Security uintptr, __StackSize uint32, __StartAddress T_beginthreadex_proc_type, __ArgList uintptr, __InitFlag uint32, __ThrdAddr uintptr) (r Tuintptr_t) {
	f := (*struct {
		f func(*TLS, uintptr) uint32
	})(unsafe.Pointer(&struct{ uintptr }{__StartAddress})).f
	var adapter = threadAdapter{threadFunc: f, tls: NewTLS(), param: __ArgList}
	adapter.token = addObject(&adapter)
	r0, _, _ := syscall.SyscallN(procCreateThread.Addr(), 0, uintptr(__StackSize), threadCallback, adapter.token, 0, 0)
	return Tuintptr_t(r0)
}

// __attribute__ ((__dllimport__)) void __attribute__((__cdecl__)) _endthreadex(unsigned _Retval) __attribute__ ((__noreturn__));
func X_endthreadex(tls *TLS, __Retval uint32) {
	tls.endthread(__Retval)
}

func Start(main func(*TLS, int32, uintptr) int32) {
	runtime.LockOSThread()
	argv := Xcalloc(nil, 1, Tsize_t((len(os.Args)+1)*int(uintptrSize)))
	if argv == 0 {
		panic("OOM")
	}

	p := argv
	for _, v := range os.Args {
		s := Xcalloc(nil, 1, Tsize_t(len(v)+1))
		if s == 0 {
			panic("OOM")
		}

		copy(unsafe.Slice((*byte)(unsafe.Pointer(s)), len(v)), v)
		*(*uintptr)(unsafe.Pointer(p)) = s
		p += uintptrSize
	}
	t := NewTLS()
	rc := main(t, int32(len(os.Args)), argv)
	Xexit(t, rc)
}

type tlsStackSlot struct {
	p  uintptr
	sz Tsize_t
}

// TLS emulates thread local storage. TLS is not safe for concurrent use by
// multiple goroutines.
type TLS struct {
	errnop    uintptr
	lastError uint32 // libkernel32.{SetLastError,GetLastError}
	stack     []tlsStackSlot
	token     uintptr

	retval uint32
	sp     int

	exited bool
}

// NewTLS returns a newly created TLS that must be eventually closed to prevent
// resource leaks.
func NewTLS() (r *TLS) {
	p := mustMalloc(Tsize_t(unsafe.Sizeof(int32(0))))
	*(*int32)(unsafe.Pointer(p)) = 0
	return &TLS{
		errnop: p,
	}
}

var procexit = dll.NewProc("exit")
var _ = procexit.Addr()

// void __attribute__((__cdecl__)) __attribute__ ((__nothrow__)) exit(int _Code) __attribute__ ((__noreturn__));
func Xexit(tls *TLS, __Code int32) {
	Xfflush(tls, Xstdout)
	Xfflush(tls, Xstderr)
	syscall.SyscallN(procexit.Addr(), uintptr(__Code))
}

func (tls *TLS) endthread(retval uint32) {
	if tls == nil || tls.exited {
		return
	}

	tls.exited = true
	tls.retval = retval
	tls.Close()
	removeObject(tls.token)
}

func (tls *TLS) SetLastError(_dwErrCode uint32) {
	if tls != nil {
		tls.lastError = _dwErrCode
	}
}

// https://github.com/golang/go/issues/41220

func (tls *TLS) GetLastError() (r uint32) {
	if tls == nil {
		return 0
	}

	return tls.lastError
}

func (tls *TLS) setErrno(n int32) {
	if __ccgo_strace {
		trc("errno<-%v", n)
	}
	if tls == nil {
		return
	}

	*(*int32)(unsafe.Pointer(tls.errnop)) = n
}

func X___errno_location(t *TLS) uintptr {
	return t.errnop
}

// int * __errno_location(void);
func X__errno_location(t *TLS) uintptr {
	return t.errnop
}

// __attribute__ ((__dllimport__)) extern int * __attribute__((__cdecl__)) _errno(void);
func X_errno(tls *TLS) (r uintptr) {
	return tls.errnop
}

func malloc(n Tsize_t) (r uintptr) {
	allocatorMu.Lock()

	defer allocatorMu.Unlock()

	if r, _ = allocator.UintptrMalloc(int(n)); r == 0 {
		panic("OOM")
	}

	return r
}

func free(p uintptr) {
	allocatorMu.Lock()

	defer allocatorMu.Unlock()

	allocator.UintptrFree(p)
}

func malloc_usable_size(p uintptr) (r Tsize_t) {
	if p == 0 {
		return 0
	}

	allocatorMu.Lock()

	defer allocatorMu.Unlock()

	return Tsize_t(memory.UintptrUsableSize(p))
}

func (tls *TLS) Close() {
	if tls == nil {
		return
	}

	// for _, v := range tls.allocas {
	// 	free(tls, v)
	// }
	for _, v := range tls.stack /* shrink diabled[:tls.sp] */ {
		free(v.p)
	}
}

func (tls *TLS) Alloc(n0 int) (r uintptr) {
	const shrinkSegment = 32
	n := Tsize_t(n0)
	if tls.sp < len(tls.stack) {
		p := tls.stack[tls.sp].p
		sz := tls.stack[tls.sp].sz
		if sz >= n /* && sz <= shrinkSegment*n */ {
			// Segment shrinking is nice to have but Tcl does some dirty hacks in coroutine
			// handling that require stability of stack addresses, out of the C execution
			// model. Disabled.
			tls.sp++
			return p
		}

		free(p)
		r = malloc(n)
		tls.stack[tls.sp] = tlsStackSlot{p: r, sz: malloc_usable_size(r)}
		tls.sp++
		return r

	}

	r = malloc(n)
	tls.stack = append(tls.stack, tlsStackSlot{p: r, sz: malloc_usable_size(r)})
	tls.sp++
	return r
}

// Free manages memory of the preceding TLS.Alloc()
func (tls *TLS) Free(n int) {
	tls.sp--
}

// VaList fills a varargs list at p with args and returns p.  The list must
// have been allocated by the caller and it must not be in Go managed memory,
// ie.  it must be pinned. The caller is responsible for freeing the list.
//
// This function supports code generated by ccgo/v4.
//
// Note: The C translated to Go varargs ABI alignment for all types is 8 on all
// architectures.
func VaList(p uintptr, args ...interface{}) (r uintptr) {
	if p&7 != 0 {
		panic("internal error")
	}

	r = p
	for _, v := range args {
		switch x := v.(type) {
		case int:
			*(*int64)(unsafe.Pointer(p)) = int64(x)
		case int32:
			*(*int64)(unsafe.Pointer(p)) = int64(x)
		case int64:
			*(*int64)(unsafe.Pointer(p)) = x
		case uint:
			*(*uint64)(unsafe.Pointer(p)) = uint64(x)
		case uint16:
			*(*uint64)(unsafe.Pointer(p)) = uint64(x)
		case uint32:
			*(*uint64)(unsafe.Pointer(p)) = uint64(x)
		case uint64:
			*(*uint64)(unsafe.Pointer(p)) = x
		case float64:
			*(*float64)(unsafe.Pointer(p)) = x
		case uintptr:
			*(*uintptr)(unsafe.Pointer(p)) = x
		default:
			sz := reflect.TypeOf(v).Size()
			copy(unsafe.Slice((*byte)(unsafe.Pointer(p)), sz), unsafe.Slice((*byte)(unsafe.Pointer((*[2]uintptr)(unsafe.Pointer(&v))[1])), sz))
			p += roundup(sz, 8)
			continue
		}
		p += 8
	}
	return r
}

func roundup(n, to uintptr) uintptr {
	if r := n % to; r != 0 {
		return n + to - r
	}

	return n
}

func mustMalloc(sz Tsize_t) (r uintptr) {
	if r = Xmalloc(nil, sz); r != 0 || sz == 0 {
		return r
	}

	panic("OOM")
}

// CString returns a pointer to a zero-terminated version of s. The caller is
// responsible for freeing the allocated memory using Xfree.
func CString(s string) (uintptr, error) {
	n := len(s)
	p := Xmalloc(nil, Tsize_t(n)+1)
	if p == 0 {
		return 0, fmt.Errorf("CString: cannot allocate %d bytes", n+1)
	}

	copy(unsafe.Slice((*byte)(unsafe.Pointer(p)), n), s)
	*(*byte)(unsafe.Pointer(p + uintptr(n))) = 0
	return p, nil
}

// GoBytes returns a byte slice from a C char* having length len bytes.
func GoBytes(s uintptr, len int) []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(s)), len)
}

// GoString returns the value of a C string at s.
func GoString(s uintptr) string {
	if s == 0 {
		return ""
	}

	var buf []byte
	for {
		b := *(*byte)(unsafe.Pointer(s))
		if b == 0 {
			return string(buf)
		}

		buf = append(buf, b)
		s++
	}
}

func X__ccgo_SyscallFP() {
	s := fmt.Sprintf("%s\nTODO syscall: function pointer", debug.Stack())
	panic(s)
}

func Bool(v bool) bool { return v }

func Bool32(b bool) int32 {
	if b {
		return 1
	}

	return 0
}

func fwrite(tls *TLS, f uintptr, b []byte) (int, error) {
	if len(b) == 0 {
		return 0, nil
	}

	n := Xfwrite(tls, uintptr(unsafe.Pointer(&b[0])), 1, Tsize_t(len(b)), f)
	if int(n) != len(b) {
		return int(n), fmt.Errorf("short write")
	}

	return len(b), nil
}

// int fprintf(FILE *stream, const char *format, ...);
func Xfprintf(tls *TLS, stream, format, args uintptr) int32 {
	n, _ := fwrite(tls, stream, printf(format, args))
	return int32(n)
}

func X__acrt_iob_func(tls *TLS, _index uint32) (r uintptr) {
	return iob[_index]
}

// extern __attribute__((__format__ (gnu_printf, 2, 0))) __attribute__ ((__nonnull__ (2))) int __attribute__((__cdecl__)) __mingw_vfprintf (FILE * __restrict__ , const char * __restrict__ , va_list) __attribute__ ((__nothrow__));
func X__mingw_vfprintf(tls *TLS, _0 uintptr, _1 uintptr, _2 Tva_list) (r int32) {
	return Xvfprintf(tls, _0, _1, _2)
}

func X__mingw_vsprintf(tls *TLS, _0 uintptr, _1 uintptr, _2 Tva_list) (r int32) {
	return Xvsprintf(tls, _0, _1, _2)
}

func X__builtin_inff(tls *TLS) float32 {
	return float32(math.Inf(1))
}

func X__builtin_nanf(tls *TLS, s uintptr) float32 {
	return float32(math.NaN())
}

func X__builtin_printf(tls *TLS, fmt uintptr, va uintptr) (r int32) {
	return Xprintf(tls, fmt, va)
}

func X__builtin_round(tls *TLS, x float64) (r float64) {
	return Xround(tls, x)
}

func X__builtin_roundf(tls *TLS, x float32) (r float32) {
	return Xroundf(tls, x)
}

func X__builtin_expect(t *TLS, exp, c long) long {
	return exp
}

func X__builtin_abort(t *TLS) {
	Xabort(t)
}

func X__builtin_abs(t *TLS, j int32) int32 {
	return Xabs(t, j)
}

func X__builtin_ctz(t *TLS, n uint32) int32 {
	return int32(mbits.TrailingZeros32(n))
}

func X__builtin_clz(t *TLS, n uint32) int32 {
	return int32(mbits.LeadingZeros32(n))
}

func X__builtin_clzl(t *TLS, n ulong) int32 {
	return int32(mbits.LeadingZeros32(n))
}

func X__builtin_clzll(t *TLS, n uint64) int32 {
	return int32(mbits.LeadingZeros64(n))
}

func X__builtin_constant_p_impl() { panic("internal error: should never be called") }

func X__builtin_copysign(t *TLS, x, y float64) float64 {
	return Xcopysign(t, x, y)
}

func X__builtin_copysignf(t *TLS, x, y float32) float32 {
	return Xcopysignf(t, x, y)
}

func X__builtin_copysignl(t *TLS, x, y float64) float64 {
	return Xcopysign(t, x, y)
}

func X__builtin_exit(t *TLS, status int32) {
	Xexit(t, status)
}

func X__builtin_fabs(t *TLS, x float64) float64 {
	return Xfabs(t, x)
}

func X__builtin_fabsf(t *TLS, x float32) float32 {
	return Xfabsf(t, x)
}

func X__builtin_free(t *TLS, ptr uintptr) {
	Xfree(t, ptr)
}

func X__builtin_huge_val(t *TLS) float64 {
	return math.Inf(1)
}

func X__builtin_huge_valf(t *TLS) float32 {
	return float32(math.Inf(1))
}

func X__builtin_inf(t *TLS) float64 {
	return math.Inf(1)
}

func X__builtin_infl(t *TLS) float64 {
	return math.Inf(1)
}

func X__builtin_malloc(t *TLS, size Tsize_t) uintptr {
	return Xmalloc(t, size)
}

func X__builtin_memcmp(t *TLS, s1, s2 uintptr, n Tsize_t) int32 {
	return Xmemcmp(t, s1, s2, n)
}

func X__builtin_nan(t *TLS, s uintptr) float64 {
	return math.NaN()
}

func X__builtin_nanl(t *TLS, s uintptr) float64 {
	return math.NaN()
}

func X__builtin_prefetch(t *TLS, addr, args uintptr) {
}

func X__builtin_strchr(t *TLS, s uintptr, c int32) uintptr {
	return Xstrchr(t, s, c)
}

func X__builtin_strcmp(t *TLS, s1, s2 uintptr) int32 {
	return Xstrcmp(t, s1, s2)
}

func X__builtin_strcpy(t *TLS, dest, src uintptr) uintptr {
	return Xstrcpy(t, dest, src)
}

func X__builtin_strlen(t *TLS, s uintptr) Tsize_t {
	return Xstrlen(t, s)
}

func X__builtin_trap(t *TLS) {
	Xabort(t)
}

func X__builtin_popcount(t *TLS, x uint32) int32 {
	return int32(mbits.OnesCount32(x))
}

// int __builtin_popcountl (unsigned long x)
func X__builtin_popcountl(t *TLS, x ulong) int32 {
	return int32(mbits.OnesCount32(x))
}

// char * __builtin___strcpy_chk (char *dest, const char *src, size_t os);
func X__builtin___strcpy_chk(t *TLS, dest, src uintptr, os Tsize_t) uintptr {
	return Xstrcpy(t, dest, src)
}

// uint16_t __builtin_bswap16 (uint32_t x)
func X__builtin_bswap16(t *TLS, x uint16) uint16 {
	return x<<8 |
		x>>8
}

// uint32_t __builtin_bswap32 (uint32_t x)
func X__builtin_bswap32(t *TLS, x uint32) uint32 {
	return x<<24 |
		x&0xff00<<8 |
		x&0xff0000>>8 |
		x>>24
}

// uint64_t __builtin_bswap64 (uint64_t x)
func X__builtin_bswap64(t *TLS, x uint64) uint64 {
	return x<<56 |
		x&0xff00<<40 |
		x&0xff0000<<24 |
		x&0xff000000<<8 |
		x&0xff00000000>>8 |
		x&0xff0000000000>>24 |
		x&0xff000000000000>>40 |
		x>>56
}

// bool __builtin_add_overflow (type1 a, type2 b, type3 *res)
func X__builtin_add_overflowInt64(t *TLS, a, b int64, res uintptr) int32 {
	r, ovf := mathutil.AddOverflowInt64(a, b)
	*(*int64)(unsafe.Pointer(res)) = r
	return Bool32(ovf)
}

// bool __builtin_add_overflow (type1 a, type2 b, type3 *res)
func X__builtin_add_overflowUint32(t *TLS, a, b uint32, res uintptr) int32 {
	r := a + b
	*(*uint32)(unsafe.Pointer(res)) = r
	return Bool32(r < a)
}

// bool __builtin_add_overflow (type1 a, type2 b, type3 *res)
func X__builtin_add_overflowUint64(t *TLS, a, b uint64, res uintptr) int32 {
	r := a + b
	*(*uint64)(unsafe.Pointer(res)) = r
	return Bool32(r < a)
}

// bool __builtin_sub_overflow (type1 a, type2 b, type3 *res)
func X__builtin_sub_overflowInt64(t *TLS, a, b int64, res uintptr) int32 {
	r, ovf := mathutil.SubOverflowInt64(a, b)
	*(*int64)(unsafe.Pointer(res)) = r
	return Bool32(ovf)
}

// bool __builtin_mul_overflow (type1 a, type2 b, type3 *res)
func X__builtin_mul_overflowInt64(t *TLS, a, b int64, res uintptr) int32 {
	r, ovf := mathutil.MulOverflowInt64(a, b)
	*(*int64)(unsafe.Pointer(res)) = r
	return Bool32(ovf)
}

// bool __builtin_mul_overflow (type1 a, type2 b, type3 *res)
func X__builtin_mul_overflowUint64(t *TLS, a, b uint64, res uintptr) int32 {
	hi, lo := mbits.Mul64(a, b)
	*(*uint64)(unsafe.Pointer(res)) = lo
	return Bool32(hi != 0)
}

// bool __builtin_mul_overflow (type1 a, type2 b, type3 *res)
func X__builtin_mul_overflowUint128(t *TLS, a, b Uint128, res uintptr) int32 {
	r, ovf := a.mulOvf(b)
	*(*Uint128)(unsafe.Pointer(res)) = r
	return Bool32(ovf)
}

func X__builtin_unreachable(t *TLS) {
	fmt.Fprintf(os.Stderr, "unrechable\n")
	os.Stderr.Sync()
	Xexit(t, 1)
}

func X__builtin_sprintf(t *TLS, str, format, args uintptr) (r int32) {
	return Xsprintf(t, str, format, args)
}

func X__builtin_memcpy(t *TLS, dest, src uintptr, n Tsize_t) (r uintptr) {
	return Xmemcpy(t, dest, src, n)
}

// void * __builtin___memcpy_chk (void *dest, const void *src, size_t n, size_t os);
func X__builtin___memcpy_chk(t *TLS, dest, src uintptr, n, os Tsize_t) (r uintptr) {
	if os != ^Tsize_t(0) && n < os {
		Xabort(t)
	}

	return Xmemcpy(t, dest, src, n)
}

func X__builtin_memset(t *TLS, s uintptr, c int32, n Tsize_t) uintptr {
	return Xmemset(t, s, c, n)
}

// void * __builtin___memset_chk (void *s, int c, size_t n, size_t os);
func X__builtin___memset_chk(t *TLS, s uintptr, c int32, n, os Tsize_t) uintptr {
	if os < n {
		Xabort(t)
	}

	return Xmemset(t, s, c, n)
}

// size_t __builtin_object_size (const void * ptr, int type)
func X__builtin_object_size(t *TLS, p uintptr, typ int32) Tsize_t {
	return ^Tsize_t(0) //TODO frontend magic
}

// int __builtin___sprintf_chk (char *s, int flag, size_t os, const char *fmt, ...);
func X__builtin___sprintf_chk(t *TLS, s uintptr, flag int32, os Tsize_t, format, args uintptr) (r int32) {
	return Xsprintf(t, s, format, args)
}

func X__builtin_isnan(t *TLS, x float64) int32 {
	return Bool32(math.IsNaN(x))
}

func X__builtin_isnanf(t *TLS, x float32) int32 {
	return Bool32(math.IsNaN(float64(x)))
}

func X__isnanl(t *TLS, arg float64) int32 {
	return X__builtin_isnanl(t, arg)
}

func X__builtin_isnanl(t *TLS, x float64) int32 {
	return Bool32(math.IsNaN(x))
}

func X__builtin_log2(t *TLS, x float64) float64 {
	return Xlog2(t, x)
}

func X__builtin___strncpy_chk(t *TLS, dest, src uintptr, n, os Tsize_t) (r uintptr) {
	if n != ^Tsize_t(0) && os < n {
		Xabort(t)
	}

	return Xstrncpy(t, dest, src, n)
}

func X__builtin___strcat_chk(t *TLS, dest, src uintptr, os Tsize_t) (r uintptr) {
	return Xstrcat(t, dest, src)
}

func X__builtin___memmove_chk(t *TLS, dest, src uintptr, n, os Tsize_t) uintptr {
	if os != ^Tsize_t(0) && os < n {
		Xabort(t)
	}

	return Xmemmove(t, dest, src, n)
}

func X__builtin_isunordered(t *TLS, a, b float64) int32 {
	return Bool32(math.IsNaN(a) || math.IsNaN(b))
}

func X__builtin_rintf(tls *TLS, x float32) (r float32) {
	return Xrintf(tls, x)
}

func X__builtin_lrintf(tls *TLS, x float32) (r long) {
	return Xlrintf(tls, x)
}

func X__builtin_lrint(tls *TLS, x float64) (r long) {
	return Xlrint(tls, x)
}

// double __builtin_fma(double x, double y, double z);
func X__builtin_fma(tls *TLS, x, y, z float64) (r float64) {
	return math.FMA(x, y, z)
}

func X__builtin_isprint(tls *TLS, c int32) (r int32) {
	return Xisprint(tls, c)
}

func X__builtin_trunc(tls *TLS, x float64) (r float64) {
	return Xtrunc(tls, x)
}

func X__builtin_hypot(tls *TLS, x float64, y float64) (r float64) {
	return Xhypot(tls, x, y)
}

func X__builtin_vsnprintf(t *TLS, str uintptr, size Tsize_t, format, va uintptr) int32 {
	return Xsnprintf(t, str, size, format, va)
}

func X__mingw_vsnprintf(tls *TLS, str uintptr, size Tsize_t, format, va uintptr) int32 {
	return Xsnprintf(tls, str, size, format, va)
}

func X__ms_vsnprintf(tls *TLS, str uintptr, size Tsize_t, format, va uintptr) int32 {
	return Xsnprintf(tls, str, size, format, va)
}

func X_vsnprintf(tls *TLS, str uintptr, size Tsize_t, format, va uintptr) int32 {
	return Xsnprintf(tls, str, size, format, va)
}

// int snprintf(char *str, size_t size, const char *format, ...);
func Xsnprintf(t *TLS, str uintptr, size Tsize_t, format, args uintptr) (r int32) {
	if __ccgo_strace {
		trc("t=%v str=%v size=%v args=%v, (%v:)", t, str, size, args, origin(2))
		defer func() { trc("-> %v", r) }()
	}
	if format == 0 {
		return 0
	}

	b := printf(format, args)
	r = int32(len(b))
	if size == 0 {
		return r
	}

	if len(b)+1 > int(size) {
		b = b[:size-1]
	}
	n := len(b)
	copy(unsafe.Slice((*byte)(unsafe.Pointer(str)), n)[:n:n], b)
	*(*byte)(unsafe.Pointer(str + uintptr(n))) = 0
	return r
}

var proc_open = dll.NewProc("_open")
var _ = proc_open.Addr()

// int open(const char * __filename, int __flags, ...)
func X_open(tls *TLS, filename uintptr, flags int32, va uintptr) (r int32) {
	var mode int32
	if va != 0 {
		mode = VaInt32(&va)
	}
	if __ccgo_strace {
		trc("filename=%q flags=%v mode=%#0o", GoString(filename), flags, mode)
		defer func() { trc(`X_open->%+v`, r) }()
	}
	r0, r1, err := syscall.SyscallN(proc_open.Addr(), filename, uintptr(flags), uintptr(mode))
	_, _ = r0, r1
	if err != 0 {
		tls.setErrno(int32(err))
	}
	return int32(r0)
}

func Xopen(tls *TLS, filename uintptr, flags int32, va uintptr) int32 {
	return X_open(tls, filename, flags, va)
}

// This version does include the zero terminator in the returned Go string.
func goWideString(p uintptr) string {
	if p == 0 {
		return ""
	}
	var w []uint16
	for {
		wc := *(*uint16)(unsafe.Pointer(p))
		p += 2
		w = append(w, wc)
		// append until U0000
		if wc == 0 {
			break
		}
	}
	s := utf16.Decode(w)
	return string(s)
}

var proc_wopen = dll.NewProc("_wopen")
var _ = proc_wopen.Addr()

// int wopen(wchar * __filename, int __flags, ...)
func X_wopen(tls *TLS, filename uintptr, flags int32, va uintptr) (r int32) {
	var mode int32
	if va != 0 {
		mode = VaInt32(&va)
	}
	if __ccgo_strace {
		trc("filename=%q flags=%v mode=%#0o", goWideString(filename), flags, mode)
		defer func() { trc(`X_wopen->%+v`, r) }()
	}
	r0, r1, err := syscall.SyscallN(proc_wopen.Addr(), filename, uintptr(flags), uintptr(mode))
	_, _ = r0, r1
	if err != 0 {
		tls.setErrno(int32(err))
	}
	return int32(r0)
}

func Xwopen(tls *TLS, filename uintptr, flags int32, va uintptr) int32 {
	return X_wopen(tls, filename, flags, va)
}

func Xread(tls *TLS, __FileHandle int32, __DstBuf uintptr, __MaxCharCount uint32) (r int32) {
	return X_read(tls, __FileHandle, __DstBuf, __MaxCharCount)
}

func Xclose(tls *TLS, __FileHandle int32) (r int32) {
	return X_close(tls, __FileHandle)
}

func Xwrite(tls *TLS, __FileHandle int32, __Buf uintptr, __MaxCharCount uint32) (r int32) {
	return X_write(tls, __FileHandle, __Buf, __MaxCharCount)
}

func Xunlink(tls *TLS, __Filename uintptr) (r int32) {
	return X_unlink(tls, __Filename)
}

func Xsetmode(tls *TLS, __FileHandle int32, __Mode int32) (r int32) {
	return X_setmode(tls, __FileHandle, __Mode)
}

func Xfileno(tls *TLS, __File uintptr) (r int32) {
	return X_fileno(tls, __File)
}

// void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
func Xqsort(tls *TLS, base uintptr, nmemb, size Tsize_t, compar uintptr) {
	sort.Sort(&sorter{
		len:  int(nmemb),
		base: base,
		sz:   uintptr(size),
		f: (*struct {
			f func(*TLS, uintptr, uintptr) int32
		})(unsafe.Pointer(&struct{ uintptr }{compar})).f,
		t: tls,
	})
}

func Xprintf(tls *TLS, format, va uintptr) int32 {
	b := printf(format, va)
	if len(b) == 0 {
		return 0
	}

	n, err := fwrite(tls, Xstdout, printf(format, va))
	if err != nil {
		return -1
	}

	return int32(n)
}

//UCRT // int __attribute__((__cdecl__)) putchar(int _Ch);
//UCRT func Xputchar(tls *TLS, __Ch int32) (r int32) {
//UCRT 	return Xputc(tls, __Ch, Xstdout)
//UCRT }

//UCRT // int __attribute__((__cdecl__)) puts(const char *_Str);
//UCRT func Xputs(tls *TLS, __Str uintptr) (r int32) {
//UCRT 	r = Xfputs(tls, __Str, Xstdout)
//UCRT 	return r + Xputc(tls, '\n', Xstdout)
//UCRT }

func Xtzset(tls *TLS) {
	X_tzset(tls)
}

func GoWideString(p uintptr) string {
	return goWideStringNZ(p)
}

func goWideStringNZ(p uintptr) string {
	if p == 0 {
		return ""
	}

	var w []uint16
	for {
		wc := *(*uint16)(unsafe.Pointer(p))
		p += 2
		if wc == 0 {
			break
		}

		w = append(w, wc)
	}
	s := utf16.Decode(w)
	return string(s)
}

func X_wgetenv(t *TLS, varname uintptr) uintptr {
	if wenvValid.Swap(true) == false {
		bootWinEnviron(t)
	}
	k := strings.ToLower(goWideStringNZ(varname))
	for _, v := range winEnviron[:len(winEnviron)-1] {
		s := strings.ToLower(goWideStringNZ(v))
		x := strings.IndexByte(s, '=')
		if s[:x] == k {
			return v
		}
	}

	return 0
}

func allocW(t *TLS, v string) (r uintptr) {
	s := utf16.Encode([]rune(v))
	p := Xcalloc(t, Tsize_t(len(s)+1), 2)
	if p == 0 {
		panic(todo(""))
	}

	r = p
	for _, v := range s {
		*(*uint16)(unsafe.Pointer(p)) = v
		p += 2
	}
	return r
}

func X_wputenv(t *TLS, envstring uintptr) int32 {
	if wenvValid.Swap(true) == false {
		bootWinEnviron(t)
	}
	s0 := goWideStringNZ(envstring)
	s := strings.ToLower(s0)
	x := strings.IndexByte(s, '=')
	k := s[:x]
	for i, v := range winEnviron[:len(winEnviron)-1] {
		s2 := strings.ToLower(goWideStringNZ(v))
		x := strings.IndexByte(s2, '=')
		if s2[:x] == k {
			Xfree(t, v)
			winEnviron[i] = allocW(t, s0)
			return 0
		}
	}

	np := allocW(t, s0)
	winEnviron = winEnviron[:len(winEnviron)-1]
	winEnviron = append(winEnviron, np, 0)
	wenviron = uintptr(unsafe.Pointer(&winEnviron[0]))
	return 0
}

func bootWinEnviron(t *TLS) {
	winEnviron = winEnviron[:0]
	for _, s := range os.Environ() {
		r := allocW(t, s)
		winEnviron = append(winEnviron, r)
	}
	wenviron = uintptr(unsafe.Pointer(&winEnviron[0]))
}

func X_copysign(t *TLS, x, y float64) float64 {
	return Xcopysign(t, x, y)
}

func Xchmod(tls *TLS, __Filename uintptr, __Mode int32) (r int32) {
	return X_chmod(tls, __Filename, __Mode)
}

func Xisatty(tls *TLS, __FileHandle int32) (r int32) {
	return X_isatty(tls, __FileHandle)
}

func AtomicLoadNUint8(ptr uintptr, memorder int32) uint8 {
	return byte(a_load_8(ptr))
}

func AtomicLoadNUint16(ptr uintptr, memorder int32) uint16 {
	return uint16(a_load_16(ptr))
}

func AtomicStoreNUint8(ptr uintptr, val uint8, memorder int32) {
	a_store_8(ptr, byte(val))
}

func AtomicStoreNUint16(ptr uintptr, val uint16, memorder int32) {
	a_store_16(ptr, val)
}

func AtomicLoadPInt32(addr uintptr) (val int32) {
	return atomic.LoadInt32((*int32)(unsafe.Pointer(addr)))
}

func AtomicLoadPInt64(addr uintptr) (val int64) {
	return atomic.LoadInt64((*int64)(unsafe.Pointer(addr)))
}

func AtomicLoadPUint32(addr uintptr) (val uint32) {
	return atomic.LoadUint32((*uint32)(unsafe.Pointer(addr)))
}

func AtomicLoadPUint64(addr uintptr) (val uint64) {
	return atomic.LoadUint64((*uint64)(unsafe.Pointer(addr)))
}

func AtomicLoadPUintptr(addr uintptr) (val uintptr) {
	return atomic.LoadUintptr((*uintptr)(unsafe.Pointer(addr)))
}

func AtomicLoadPFloat32(addr uintptr) (val float32) {
	return math.Float32frombits(atomic.LoadUint32((*uint32)(unsafe.Pointer(addr))))
}

func AtomicLoadPFloat64(addr uintptr) (val float64) {
	return math.Float64frombits(atomic.LoadUint64((*uint64)(unsafe.Pointer(addr))))
}

func AtomicStorePInt32(addr uintptr, val int32) {
	atomic.StoreInt32((*int32)(unsafe.Pointer(addr)), val)
}

func AtomicStorePInt64(addr uintptr, val int64) {
	atomic.StoreInt64((*int64)(unsafe.Pointer(addr)), val)
}

func AtomicStorePUint32(addr uintptr, val uint32) {
	atomic.StoreUint32((*uint32)(unsafe.Pointer(addr)), val)
}

func AtomicStorePUint64(addr uintptr, val uint64) {
	atomic.StoreUint64((*uint64)(unsafe.Pointer(addr)), val)
}

func AtomicStorePUintptr(addr uintptr, val uintptr) {
	atomic.StoreUintptr((*uintptr)(unsafe.Pointer(addr)), val)
}

func AtomicStorePFloat32(addr uintptr, val float32) {
	atomic.StoreUint32((*uint32)(unsafe.Pointer(addr)), math.Float32bits(val))
}

func AtomicStorePFloat64(addr uintptr, val float64) {
	atomic.StoreUint64((*uint64)(unsafe.Pointer(addr)), math.Float64bits(val))
}

// int _snwprintf(
//
//	wchar_t *buffer,
//	size_t count,
//	const wchar_t *format [,
//	argument] ...
//
// );
func X_snwprintf(tls *TLS, buffer uintptr, count Tsize_t, format, va uintptr) int32 {
	panic(todo(""))
}

func Xsnwprintf(tls *TLS, buffer uintptr, count Tsize_t, format, va uintptr) int32 {
	return X_snwprintf(tls, buffer, count, format, va)
}

// int _vscprintf(const char *format, va_list va);
func X_vscprintf(t *TLS, format uintptr, va uintptr) int32 {
	return int32(len(printf(format, va)))
}

func X_sscanf(t *TLS, str, format, va uintptr) int32 {
	return scanf(strings.NewReader(GoString(str)), format, va)
}

// int sscanf(const char *str, const char *format, ...);
func Xsscanf(t *TLS, str, format, va uintptr) int32 {
	return X_sscanf(t, str, format, va)
}

// pid_t getpid(void);
func Xgetpid(t *TLS) int32 {
	return int32(os.Getpid())
}

// void __assert_fail(const char * assertion, const char * file, unsigned int line, const char * function);
func X__assert_fail(t *TLS, assertion, file uintptr, line uint32, function uintptr) {
	if __ccgo_strace {
		trc("t=%v file=%v line=%v function=%v, (%v:)", t, file, line, function, origin(2))
	}
	fmt.Fprintf(os.Stderr, "assertion failure: %s:%d.%s: %s\n", GoString(file), line, GoString(function), GoString(assertion))
	os.Stderr.Sync()
	Xexit(t, 1)
}

// unsigned long long strtoull(const char *nptr, char **endptr, int base);
//UCRT func Xstrtoull(t *TLS, nptr, endptr uintptr, base int32) uint64 {
//UCRT 	var s uintptr = nptr
//UCRT 	var acc uint64
//UCRT 	var c byte
//UCRT 	var cutoff uint64
//UCRT 	var neg int32
//UCRT 	var any int32
//UCRT 	var cutlim int32
//UCRT
//UCRT 	/*
//UCRT 	 * Skip white space and pick up leading +/- sign if any.
//UCRT 	 * If base is 0, allow 0x for hex and 0 for octal, else
//UCRT 	 * assume decimal; if base is already 16, allow 0x.
//UCRT 	 */
//UCRT 	for {
//UCRT 		c = *(*byte)(unsafe.Pointer(s))
//UCRT 		PostIncUintptr(&s, 1)
//UCRT 		var sp = strings.TrimSpace(string(c))
//UCRT 		if len(sp) > 0 {
//UCRT 			break
//UCRT 		}
//UCRT 	}
//UCRT
//UCRT 	if c == '-' {
//UCRT 		neg = 1
//UCRT 		c = *(*byte)(unsafe.Pointer(s))
//UCRT 		PostIncUintptr(&s, 1)
//UCRT 	} else if c == '+' {
//UCRT 		c = *(*byte)(unsafe.Pointer(s))
//UCRT 		PostIncUintptr(&s, 1)
//UCRT 	}
//UCRT
//UCRT 	sp := *(*byte)(unsafe.Pointer(s))
//UCRT
//UCRT 	if (base == 0 || base == 16) &&
//UCRT 		c == '0' && (sp == 'x' || sp == 'X') {
//UCRT 		PostIncUintptr(&s, 1)
//UCRT 		c = *(*byte)(unsafe.Pointer(s)) //s[1];
//UCRT 		PostIncUintptr(&s, 1)
//UCRT 		base = 16
//UCRT 	}
//UCRT 	if base == 0 {
//UCRT 		if c == '0' {
//UCRT 			base = 0
//UCRT 		} else {
//UCRT 			base = 10
//UCRT 		}
//UCRT 	}
//UCRT
//UCRT 	cutoff = math.MaxUint64 / uint64(base)
//UCRT 	cutlim = int32(math.MaxUint64 % uint64(base))
//UCRT
//UCRT 	acc = 0
//UCRT 	any = 0
//UCRT
//UCRT 	for {
//UCRT 		var cs = string(c)
//UCRT 		if unicode.IsDigit([]rune(cs)[0]) {
//UCRT 			c -= '0'
//UCRT 		} else if unicode.IsLetter([]rune(cs)[0]) {
//UCRT 			if unicode.IsUpper([]rune(cs)[0]) {
//UCRT 				c -= 'A' - 10
//UCRT 			} else {
//UCRT 				c -= 'a' - 10
//UCRT 			}
//UCRT 		} else {
//UCRT 			break
//UCRT 		}
//UCRT
//UCRT 		if int32(c) >= base {
//UCRT 			break
//UCRT 		}
//UCRT 		if any < 0 || acc > cutoff || (acc == cutoff && int32(c) > cutlim) {
//UCRT 			any = -1
//UCRT
//UCRT 		} else {
//UCRT 			any = 1
//UCRT 			acc *= uint64(base)
//UCRT 			acc += uint64(c)
//UCRT 		}
//UCRT
//UCRT 		c = *(*byte)(unsafe.Pointer(s))
//UCRT 		PostIncUintptr(&s, 1)
//UCRT 	}
//UCRT
//UCRT 	if any < 0 {
//UCRT 		acc = math.MaxUint64
//UCRT 		t.setErrno(ERANGE)
//UCRT 	} else if neg == 1 {
//UCRT 		acc = -acc
//UCRT 	}
//UCRT
//UCRT 	if endptr != 0 {
//UCRT 		if any == 1 {
//UCRT 			PostDecUintptr(&s, 1)
//UCRT 			AssignPtrUintptr(endptr, s)
//UCRT 		} else {
//UCRT 			AssignPtrUintptr(endptr, nptr)
//UCRT 		}
//UCRT 	}
//UCRT 	return acc
//UCRT }

// int _stat32i64(const char *path, struct _stat32i64 *buffer);
//UCRT func X_stat64i32(t *TLS, path uintptr, buffer uintptr) int32 {
//UCRT 	panic(todo(""))
//UCRT }

func X__mingw_strtod(t *TLS, s uintptr, p uintptr) float64 {
	return Xstrtod(t, s, p)
}

// int vsscanf(const char *str, const char *format, va_list ap);
func X__ms_vsscanf(t *TLS, str, format, ap uintptr) int32 {
	panic(todo(""))
}

// int vsscanf(const char *str, const char *format, va_list ap);
func X__mingw_vsscanf(t *TLS, str, format, ap uintptr) int32 {
	return Xsscanf(t, str, format, ap)
}

// unsigned int _set_abort_behavior(
//
//	unsigned int flags,
//	unsigned int mask
//
// );
//UCRT func X_set_abort_behavior(t *TLS, _ ...interface{}) uint32 {
//UCRT 	panic(todo(""))
//UCRT }

// double atof(const char *nptr);
func Xatof(t *TLS, nptr uintptr) float64 {
	n, _ := strconv.ParseFloat(GoString(nptr), 64)
	return n
}

func X__builtin_snprintf(t *TLS, str uintptr, size Tsize_t, format, args uintptr) int32 {
	return Xsnprintf(t, str, size, format, args)
}

// int _snprintf(char *str, size_t size, const char *format, ...);
func X_snprintf(t *TLS, str uintptr, size Tsize_t, format, args uintptr) int32 {
	return Xsnprintf(t, str, size, format, args)
}

// intptr_t _findfirst64i32(const char *filespec, struct _finddata64i32_t *fileinfo);
//UCRT func X_findfirst64i32(t *TLS, filespec, fileinfo uintptr) Tintptr_t {
//UCRT 	panic(todo(""))
//UCRT }

// int _findnext64i32(intptr_t handle, struct _finddata64i32_t *fileinfo);
//UCRT func X_findnext64i32(t *TLS, handle Tintptr_t, fileinfo uintptr) int32 {
//UCRT 	panic(todo(""))
//UCRT }

func X__isnan(t *TLS, x float64) int32 {
	return Bool32(math.IsNaN(x))
}

// int _stati64(char *path, struct _stati64 *buffer);
func X_stati64(t *TLS, path, buffer uintptr) int32 {
	return X_stat64(t, path, buffer)
}

// int _fstati64(int fd, struct _stati64 *buffer);
func X_fstati64(t *TLS, fd int32, buffer uintptr) int32 {
	return X_fstat64(t, fd, buffer)
}

func X_strcmpi(tls *TLS, __Str1 uintptr, __Str2 uintptr) (r int32) {
	if __ccgo_strace {
		trc("_Str1=%+v _Str2=%+v", __Str1, __Str2)
		defer func() { trc(`X_strcmpi->%+v`, r) }()
	}
	r0, r1, err := syscall.SyscallN(proc_stricmp.Addr(), __Str1, __Str2)
	if err != 0 {
		if __ccgo_strace {
			trc(`r0=%v r1=%v err=%v`, r0, r1, err)
		}
		tls.setErrno(int32(err))
	}
	return int32(r0)
}

// ------------------------------------------------------------------------ (A)
// https://chromium.googlesource.com/external/github.com/kripken/emscripten/+/refs/tags/1.2.9/system/lib/libc/stdlib/strtod.c
//
//	/*
//	 * strtod.c --
//	 *
//	 *	Source code for the "strtod" library procedure.
//	 *
//	 * Copyright (c) 1988-1993 The Regents of the University of California.
//	 * Copyright (c) 1994 Sun Microsystems, Inc.
//	 *
//	 * Permission to use, copy, modify, and distribute this
//	 * software and its documentation for any purpose and without
//	 * fee is hereby granted, provided that the above copyright
//	 * notice appear in all copies.  The University of California
//	 * makes no representations about the suitability of this
//	 * software for any purpose.  It is provided "as is" without
//	 * express or implied warranty.
//	 *
//	 * RCS: @(#) $Id$
//	 *
//	 * Taken from http://svn.ruby-lang.org/repos/ruby/branches/ruby_1_8/missing/strtod.c
//	 */
//
// C documentation
//
//	/*
//	 *----------------------------------------------------------------------
//	 *
//	 * strtod --
//	 *
//	 *	This procedure converts a floating-point number from an ASCII
//	 *	decimal representation to internal double-precision format.
//	 *
//	 * Results:
//	 *	The return value is the double-precision floating-point
//	 *	representation of the characters in string.  If endPtr isn't
//	 *	NULL, then *endPtr is filled in with the address of the
//	 *	next character after the last one that was part of the
//	 *	floating-point number.
//	 *
//	 * Side effects:
//	 *	None.
//	 *
//	 *----------------------------------------------------------------------
//	 */
func Xstrtod(tls *TLS, string1 uintptr, endPtr uintptr) (r float64) {
	/* If non-NULL, store terminating character's
	 * address here. */
	var c, decPt, exp, expSign, frac1, frac2, fracExp, mantSize, sign, v1, v2 int32
	var d, p, pExp uintptr
	var dblExp, fraction float64
	_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = c, d, dblExp, decPt, exp, expSign, frac1, frac2, fracExp, fraction, mantSize, p, pExp, sign, v1, v2
	expSign = FALSE
	exp = 0     /* Exponent read from "EX" field. */
	fracExp = 0 /* Temporarily holds location of exponent
	 * in string. */
	/*
	 * Strip off leading blanks and check for a sign.
	 */
	p = string1
	for {
		v1 = int32(*(*int8)(unsafe.Pointer(p)))
		v2 = BoolInt32(v1 == int32(' ') || uint32(v1)-uint32('\t') < uint32(5))
		goto _3
	_3:
		if !(v2 != 0) {
			break
		}
		p += uintptr(1)
	}
	if int32(*(*int8)(unsafe.Pointer(p))) == int32('-') {
		sign = int32(TRUE)
		p += uintptr(1)
	} else {
		if int32(*(*int8)(unsafe.Pointer(p))) == int32('+') {
			p += uintptr(1)
		}
		sign = FALSE
	}
	/*
	 * Count the number of digits in the mantissa (including the decimal
	 * point), and also locate the decimal point.
	 */
	decPt = -int32(1)
	mantSize = 0
	for {
		c = int32(*(*int8)(unsafe.Pointer(p)))
		if !(BoolInt32(uint32(c)-Uint32FromUint8('0') < Uint32FromInt32(10)) != 0) {
			if c != int32('.') || decPt >= 0 {
				break
			}
			decPt = mantSize
		}
		p += uintptr(1)
		goto _4
	_4:
		;
		mantSize += int32(1)
	}
	/*
	 * Now suck up the digits in the mantissa.  Use two integers to
	 * collect 9 digits each (this is faster than using floating-point).
	 * If the mantissa has more than 18 digits, ignore the extras, since
	 * they can't affect the value anyway.
	 */
	pExp = p
	p -= uintptr(mantSize)
	if decPt < 0 {
		decPt = mantSize
	} else {
		mantSize -= int32(1) /* One of the digits was the point. */
	}
	if mantSize > int32(18) {
		fracExp = decPt - int32(18)
		mantSize = int32(18)
	} else {
		fracExp = decPt - mantSize
	}
	if mantSize == 0 {
		fraction = float64(0)
		p = string1
		goto done
	} else {
		frac1 = 0
		for {
			if !(mantSize > int32(9)) {
				break
			}
			c = int32(*(*int8)(unsafe.Pointer(p)))
			p += uintptr(1)
			if c == int32('.') {
				c = int32(*(*int8)(unsafe.Pointer(p)))
				p += uintptr(1)
			}
			frac1 = int32(10)*frac1 + (c - int32('0'))
			goto _5
		_5:
			;
			mantSize -= int32(1)
		}
		frac2 = 0
		for {
			if !(mantSize > 0) {
				break
			}
			c = int32(*(*int8)(unsafe.Pointer(p)))
			p += uintptr(1)
			if c == int32('.') {
				c = int32(*(*int8)(unsafe.Pointer(p)))
				p += uintptr(1)
			}
			frac2 = int32(10)*frac2 + (c - int32('0'))
			goto _6
		_6:
			;
			mantSize -= int32(1)
		}
		fraction = float64(1e+09)*float64(float64(frac1)) + float64(float64(frac2))
	}
	/*
	 * Skim off the exponent.
	 */
	p = pExp
	if int32(*(*int8)(unsafe.Pointer(p))) == int32('E') || int32(*(*int8)(unsafe.Pointer(p))) == int32('e') {
		p += uintptr(1)
		if int32(*(*int8)(unsafe.Pointer(p))) == int32('-') {
			expSign = int32(TRUE)
			p += uintptr(1)
		} else {
			if int32(*(*int8)(unsafe.Pointer(p))) == int32('+') {
				p += uintptr(1)
			}
			expSign = FALSE
		}
		for BoolInt32(uint32(*(*int8)(unsafe.Pointer(p)))-uint32('0') < uint32(10)) != 0 {
			exp = exp*int32(10) + (int32(*(*int8)(unsafe.Pointer(p))) - int32('0'))
			p += uintptr(1)
		}
	}
	if expSign != 0 {
		exp = fracExp - exp
	} else {
		exp = fracExp + exp
	}
	/*
	 * Generate a floating-point number that represents the exponent.
	 * Do this by processing the exponent one bit at a time to combine
	 * many powers of 2 of 10. Then combine the exponent with the
	 * fraction.
	 */
	if exp < 0 {
		expSign = int32(TRUE)
		exp = -exp
	} else {
		expSign = FALSE
	}
	if exp > maxExponent {
		exp = maxExponent
		*(*int32)(unsafe.Pointer(X__errno_location(tls))) = int32(ERANGE)
	}
	dblExp = float64(1)
	d = uintptr(unsafe.Pointer(&powersOf10))
	for {
		if !(exp != 0) {
			break
		}
		if exp&int32(01) != 0 {
			dblExp *= *(*float64)(unsafe.Pointer(d))
		}
		goto _7
	_7:
		;
		exp >>= int32(1)
		d += uintptr(1) * 8
	}
	if expSign != 0 {
		fraction /= dblExp
	} else {
		fraction *= dblExp
	}
	goto done
done:
	;
	if endPtr != UintptrFromInt32(0) {
		*(*uintptr)(unsafe.Pointer(endPtr)) = p
	}
	if sign != 0 {
		return -fraction
	}
	return fraction
}

var maxExponent = int32(511) /* Largest possible base 10 exponent.  Any
 * exponent larger than this will already
 * produce underflow or overflow, so there's
 * no need to worry about additional digits.
 */

var powersOf10 = [9]float64{
	0: float64(10),
	1: float64(100),
	2: float64(10000),
	3: float64(1e+08),
	4: float64(1e+16),
	5: float64(1e+32),
	6: float64(1e+64),
	7: float64(1e+128),
	8: float64(1e+256),
}

// ------------------------------------------------------------------------ (Z)
