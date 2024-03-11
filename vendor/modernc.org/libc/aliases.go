// Copyright 2024 The Libc Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && amd64

package libc // import "modernc.org/libc"

func X__vm_wait(tls *TLS) {}

// static volatile int *const dummy_lockptr = 0;
//
// weak_alias(dummy_lockptr, __atexit_lockptr);
// weak_alias(dummy_lockptr, __bump_lockptr);
// weak_alias(dummy_lockptr, __sem_open_lockptr);
var X__atexit_lockptr int32
var X__bump_lockptr int32
var X__sem_open_lockptr int32

// static int dummy(int fd)
//
//	{
//		return fd;
//	}
//
// weak_alias(dummy, __aio_close);
func X__aio_close(tls *TLS, fd int32) int32 {
	return fd
}

func Xfread(tls *TLS, destv uintptr, size Tsize_t, nmemb Tsize_t, f uintptr) (r Tsize_t) {
	return Xfread_unlocked(tls, destv, size, nmemb, f)
}

func Xferror(tls *TLS, f uintptr) (r int32) {
	return Xferror_unlocked(tls, f)
}

func Xfwrite(tls *TLS, src uintptr, size Tsize_t, nmemb Tsize_t, f uintptr) (r Tsize_t) {
	return Xfwrite_unlocked(tls, src, size, nmemb, f)
}

func Xfileno(tls *TLS, f uintptr) (r int32) {
	return Xfileno_unlocked(tls, f)
}

func Xtzset(tls *TLS) {
	___tzset(tls)
}

func Xfflush(tls *TLS, f uintptr) (r1 int32) {
	return Xfflush_unlocked(tls, f)
}

func Xputc(tls *TLS, c int32, f uintptr) (r int32) {
	return X_IO_putc(tls, c, f)
}

func Xfgets(tls *TLS, s uintptr, n int32, f uintptr) (r uintptr) {
	return Xfgets_unlocked(tls, s, n, f)
}

type DIR = TDIR

const DT_DETACHED = _DT_DETACHED

const DT_EXITING = _DT_EXITING

const DT_JOINABLE = _DT_JOINABLE

type FILE = TFILE

type HEADER = THEADER

func X__inet_aton(tls *TLS, s0 uintptr, dest uintptr) (r int32) {
	return Xinet_aton(tls, s0, dest)
}

func X__isalnum_l(tls *TLS, c int32, l Tlocale_t) (r int32) {
	return Xisalnum_l(tls, c, l)
}

func X__isalpha_l(tls *TLS, c int32, l Tlocale_t) (r int32) {
	return Xisalpha_l(tls, c, l)
}

func X__isdigit_l(tls *TLS, c int32, l Tlocale_t) (r int32) {
	return Xisdigit_l(tls, c, l)
}

func X__islower_l(tls *TLS, c int32, l Tlocale_t) (r int32) {
	return Xislower_l(tls, c, l)
}

func X__isprint_l(tls *TLS, c int32, l Tlocale_t) (r int32) {
	return Xisprint_l(tls, c, l)
}

func X__isupper_l(tls *TLS, c int32, l Tlocale_t) (r int32) {
	return Xisupper_l(tls, c, l)
}

func X__isxdigit_l(tls *TLS, c int32, l Tlocale_t) (r int32) {
	return Xisxdigit_l(tls, c, l)
}

func X__strncasecmp_l(tls *TLS, l uintptr, r uintptr, n Tsize_t, loc Tlocale_t) (r1 int32) {
	return Xstrncasecmp_l(tls, l, r, n, loc)
}

func Xcfsetospeed(tls *TLS, tio uintptr, speed Tspeed_t) (r int32) {
	return Xcfsetspeed(tls, tio, speed)
}

func Xfcntl64(tls *TLS, fd int32, cmd int32, va uintptr) (r int32) {
	return Xfcntl(tls, fd, cmd, va)
}

func Xfeof(tls *TLS, f uintptr) (r int32) {
	return Xfeof_unlocked(tls, f)
}

func Xfopen64(tls *TLS, filename uintptr, mode uintptr) (r uintptr) {
	return Xfopen(tls, filename, mode)
}

func Xfputs(tls *TLS, s uintptr, f uintptr) (r int32) {
	return Xfputs_unlocked(tls, s, f)
}

func Xfscanf(tls *TLS, f uintptr, fmt uintptr, va uintptr) (r int32) {
	return X__isoc99_fscanf(tls, f, fmt, va)
}

func Xfstat64(tls *TLS, fd int32, st uintptr) (r int32) {
	return Xfstat(tls, fd, st)
}

func Xftruncate64(tls *TLS, fd int32, length Toff_t) (r int32) {
	return Xftruncate(tls, fd, length)
}

func Xgetrlimit64(tls *TLS, resource int32, rlim uintptr) (r int32) {
	return Xgetrlimit(tls, resource, rlim)
}

func Xlseek64(tls *TLS, fd int32, offset Toff_t, whence int32) (r Toff_t) {
	return Xlseek(tls, fd, offset, whence)
}

func Xlstat64(tls *TLS, path uintptr, buf uintptr) (r int32) {
	return Xlstat(tls, path, buf)
}

func Xmkstemp64(tls *TLS, template uintptr) (r int32) {
	return Xmkstemp(tls, template)
}

func Xmkstemps64(tls *TLS, template uintptr, len1 int32) (r int32) {
	return Xmkstemps(tls, template, len1)
}

func Xmmap64(tls *TLS, start uintptr, len1 Tsize_t, prot int32, flags int32, fd int32, off Toff_t) (r uintptr) {
	return Xmmap(tls, start, len1, prot, flags, fd, off)
}

func Xopen64(tls *TLS, filename uintptr, flags int32, va uintptr) (r int32) {
	return Xopen(tls, filename, flags, va)
}

func Xreaddir64(tls *TLS, dir uintptr) (r uintptr) {
	return Xreaddir(tls, dir)
}

func Xsetrlimit64(tls *TLS, resource int32, rlim uintptr) (r int32) {
	return Xsetrlimit(tls, resource, rlim)
}

func Xsscanf(tls *TLS, s uintptr, fmt uintptr, va uintptr) (r int32) {
	return X__isoc99_sscanf(tls, s, fmt, va)
}

func Xstat64(tls *TLS, path uintptr, buf uintptr) (r int32) {
	return Xstat(tls, path, buf)
}

func Xvfscanf(tls *TLS, f uintptr, fmt uintptr, ap Tva_list) (r int32) {
	return X__isoc99_vfscanf(tls, f, fmt, ap)
}

func Xvsscanf(tls *TLS, s uintptr, fmt uintptr, ap Tva_list) (r int32) {
	return X__isoc99_sscanf(tls, s, fmt, ap)
}
