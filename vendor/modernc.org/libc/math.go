// Copyright 2024 The Libc Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !(linux && (amd64 || arm64 || loong64))

package libc // import "modernc.org/libc"

import (
	"math"
	"unsafe"
)

func Xsin(t *TLS, x float64) float64 {
	return math.Sin(x)
}

func Xsinf(t *TLS, x float32) float32 {
	return float32(math.Sin(float64(x)))
}

func Xsinh(t *TLS, x float64) float64 {
	return math.Sinh(x)
}

func Xsinhf(t *TLS, x float32) float32 {
	return float32(math.Sinh(float64(x)))
}

func Xcos(t *TLS, x float64) float64 {
	return math.Cos(x)
}

func Xcosf(t *TLS, x float32) float32 {
	return float32(math.Cos(float64(x)))
}

func Xcosh(t *TLS, x float64) float64 {
	return math.Cosh(x)
}

func Xcoshf(t *TLS, x float32) float32 {
	return float32(math.Cosh(float64(x)))
}

func Xtan(t *TLS, x float64) float64 {
	return math.Tan(x)
}

func Xtanf(t *TLS, x float32) float32 {
	return float32(math.Tan(float64(x)))
}

func Xtanh(t *TLS, x float64) float64 {
	return math.Tanh(x)
}

func Xtanhf(t *TLS, x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func Xasin(t *TLS, x float64) float64 {
	return math.Asin(x)
}

func Xasinf(t *TLS, x float32) float32 {
	return float32(math.Asin(float64(x)))
}

func Xasinh(t *TLS, x float64) float64 {
	return math.Asinh(x)
}

func Xasinhf(t *TLS, x float32) float32 {
	return float32(math.Asinh(float64(x)))
}

func Xacos(t *TLS, x float64) float64 {
	return math.Acos(x)
}

func Xacosf(t *TLS, x float32) float32 {
	return float32(math.Acos(float64(x)))
}

func Xacosh(t *TLS, x float64) float64 {
	return math.Acosh(x)
}

func Xacoshf(t *TLS, x float32) float32 {
	return float32(math.Acosh(float64(x)))
}

func Xatan(t *TLS, x float64) float64 {
	return math.Atan(x)
}

func Xatanf(t *TLS, x float32) float32 {
	return float32(math.Atan(float64(x)))
}

func Xatan2(t *TLS, x, y float64) float64 {
	return math.Atan2(x, y)
}

func Xatan2f(t *TLS, x, y float32) float32 {
	return float32(math.Atan2(float64(x), float64(y)))
}

func Xatanh(t *TLS, x float64) float64 {
	return math.Atanh(x)
}

func Xatanhf(t *TLS, x float32) float32 {
	return float32(math.Atanh(float64(x)))
}

func Xexp(t *TLS, x float64) float64 {
	return math.Exp(x)
}

func Xexpf(t *TLS, x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func Xfabs(t *TLS, x float64) float64 {
	return math.Abs(x)
}

func Xfabsf(t *TLS, x float32) float32 {
	return float32(math.Abs(float64(x)))
}

func Xlog(t *TLS, x float64) float64 {
	return math.Log(x)
}

func Xlogf(t *TLS, x float32) float32 {
	return float32(math.Log(float64(x)))
}

func Xlog10(t *TLS, x float64) float64 {
	return math.Log10(x)
}

func Xlog10f(t *TLS, x float32) float32 {
	return float32(math.Log10(float64(x)))
}

func Xlog2(t *TLS, x float64) float64 {
	return math.Log2(x)
}

func Xlog2f(t *TLS, x float32) float32 {
	return float32(math.Log2(float64(x)))
}

func Xpow(t *TLS, x, y float64) float64 {
	r := math.Pow(x, y)
	if x > 0 && r == 1 && y >= -1.0000000000000000715e-18 && y < -1e-30 {
		r = 0.9999999999999999
	}
	return r
}

func Xpowf(t *TLS, x, y float32) float32 {
	return float32(math.Pow(float64(x), float64(y)))
}

func Xsqrt(t *TLS, x float64) float64 {
	return math.Sqrt(x)
}

func Xsqrtf(t *TLS, x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func Xround(t *TLS, x float64) float64 {
	return math.Round(x)
}

func Xroundf(t *TLS, x float32) float32 {
	return float32(math.Round(float64(x)))
}

func Xceil(t *TLS, x float64) float64 {
	return math.Ceil(x)
}

func Xceilf(t *TLS, x float32) float32 {
	return float32(math.Ceil(float64(x)))
}

func Xfloor(t *TLS, x float64) float64 {
	return math.Floor(x)
}

func Xfloorf(t *TLS, x float32) float32 {
	return float32(math.Floor(float64(x)))
}

func Xcopysign(t *TLS, x, y float64) float64 {
	return math.Copysign(x, y)
}

func Xcopysignf(t *TLS, x, y float32) float32 {
	return float32(math.Copysign(float64(x), float64(y)))
}

func Xfmod(t *TLS, x, y float64) float64 {
	return math.Mod(x, y)
}

func Xfmodf(t *TLS, x, y float32) float32 {
	return float32(math.Mod(float64(x), float64(y)))
}

func Xhypot(t *TLS, x, y float64) float64 {
	return math.Hypot(x, y)
}

func Xhypotf(t *TLS, x, y float32) float32 {
	return float32(math.Hypot(float64(x), float64(y)))
}

func Xisnan(t *TLS, x float64) int32 {
	return Bool32(math.IsNaN(x))
}

func Xisnanf(t *TLS, x float32) int32 {
	return Bool32(math.IsNaN(float64(x)))
}

func Xisnanl(t *TLS, x float64) int32 {
	return Bool32(math.IsNaN(x))
}

func Xldexp(t *TLS, x float64, exp int32) float64 {
	return math.Ldexp(x, int(exp))
}

func Xtrunc(t *TLS, x float64) float64 {
	return math.Trunc(x)
}

func Xtruncf(t *TLS, x float32) float32 {
	return float32(math.Trunc(float64(x)))
}

func Xfrexp(t *TLS, x float64, exp uintptr) float64 {
	f, e := math.Frexp(x)
	*(*int32)(unsafe.Pointer(exp)) = int32(e)
	return f
}

func Xfrexpf(t *TLS, x float32, exp uintptr) float32 {
	f, e := math.Frexp(float64(x))
	*(*int32)(unsafe.Pointer(exp)) = int32(e)
	return float32(f)
}

func Xmodf(t *TLS, x float64, iptr uintptr) float64 {
	i, f := math.Modf(x)
	*(*float64)(unsafe.Pointer(iptr)) = i
	return f
}

func Xmodff(t *TLS, x float32, iptr uintptr) float32 {
	i, f := math.Modf(float64(x))
	*(*float32)(unsafe.Pointer(iptr)) = float32(i)
	return float32(f)
}

var _toint5 = Float32FromInt32(1) / Float32FromFloat32(1.1920928955078125e-07)

func Xrintf(tls *TLS, x float32) (r float32) {
	bp := tls.Alloc(16)
	defer tls.Free(16)
	var e, s int32
	var y float32
	var v1 float32
	var _ /* u at bp+0 */ struct {
		Fi [0]uint32
		Ff float32
	}
	_, _, _, _ = e, s, y, v1
	*(*struct {
		Fi [0]uint32
		Ff float32
	})(unsafe.Pointer(bp)) = struct {
		Fi [0]uint32
		Ff float32
	}{}
	*(*float32)(unsafe.Pointer(bp)) = x
	e = int32(*(*uint32)(unsafe.Pointer(bp)) >> int32(23) & uint32(0xff))
	s = int32(*(*uint32)(unsafe.Pointer(bp)) >> int32(31))
	if e >= Int32FromInt32(0x7f)+Int32FromInt32(23) {
		return x
	}
	if s != 0 {
		y = x - _toint5 + _toint5
	} else {
		y = x + _toint5 - _toint5
	}
	if y == Float32FromInt32(0) {
		if s != 0 {
			v1 = -Float32FromFloat32(0)
		} else {
			v1 = Float32FromFloat32(0)
		}
		return v1
	}
	return y
}

func Xlrintf(tls *TLS, x float32) (r long) {
	return long(Xrintf(tls, x))
}

func Xlrint(tls *TLS, x float64) (r long) {
	return long(Xrint(tls, x))
}

func Xrint(tls *TLS, x float64) float64 {
	switch {
	case x == 0: // also +0 and -0
		return 0
	case math.IsInf(x, 0), math.IsNaN(x):
		return x
	case x >= math.MinInt64 && x <= math.MaxInt64 && float64(int64(x)) == x:
		return x
	case x >= 0:
		return math.Floor(x + 0.5)
	default:
		return math.Ceil(x - 0.5)
	}
}
