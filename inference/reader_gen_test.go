// GENERATED CODE, DO NOT EDIT

package inference

import (
	"os"
	"path"
	"testing"
)

func checkComplex128(data []complex128) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != complex(float64(k), 0) {
			return false
		}
	}
	return true
}
func checkComplex64(data []complex64) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != complex(float32(k), 0) {
			return false
		}
	}
	return true
}
func checkFloat64(data []float64) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != float64(k) {
			return false
		}
	}
	return true
}
func checkFloat32(data []float32) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != float32(k) {
			return false
		}
	}
	return true
}
func checkUint64(data []uint64) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != uint64(k) {
			return false
		}
	}
	return true
}
func checkUint32(data []uint32) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != uint32(k) {
			return false
		}
	}
	return true
}
func checkUint16(data []uint16) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != uint16(k) {
			return false
		}
	}
	return true
}
func checkUint8(data []uint8) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != uint8(k) {
			return false
		}
	}
	return true
}
func checkInt64(data []int64) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != int64(k) {
			return false
		}
	}
	return true
}
func checkInt32(data []int32) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != int32(k) {
			return false
		}
	}
	return true
}
func checkInt16(data []int16) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != int16(k) {
			return false
		}
	}
	return true
}
func checkInt8(data []int8) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != int8(k) {
			return false
		}
	}
	return true
}
func checkBytes(data []byte) bool {
	for k := 0; k < len(data); k++ {
		if data[k] != byte(k) {
			return false
		}
	}
	return true
}
func TestReadComplex128(t *testing.T) {

	files := getFlist("c16")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetComplex128()
		if err != nil {
			panic(err)
		}

		if !checkComplex128(data) {
			t.Fail()
		}
	}
}

func TestReadComplex64(t *testing.T) {

	files := getFlist("c8")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetComplex64()
		if err != nil {
			panic(err)
		}

		if !checkComplex64(data) {
			t.Fail()
		}
	}
}

func TestReadFloat64(t *testing.T) {

	files := getFlist("f8")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetFloat64()
		if err != nil {
			panic(err)
		}

		if !checkFloat64(data) {
			t.Fail()
		}
	}
}

func TestReadFloat32(t *testing.T) {

	files := getFlist("f4")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetFloat32()
		if err != nil {
			panic(err)
		}

		if !checkFloat32(data) {
			t.Fail()
		}
	}
}

func TestReadUint64(t *testing.T) {

	files := getFlist("u8")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetUint64()
		if err != nil {
			panic(err)
		}

		if !checkUint64(data) {
			t.Fail()
		}
	}
}

func TestReadUint32(t *testing.T) {

	files := getFlist("u4")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetUint32()
		if err != nil {
			panic(err)
		}

		if !checkUint32(data) {
			t.Fail()
		}
	}
}

func TestReadUint16(t *testing.T) {

	files := getFlist("u2")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetUint16()
		if err != nil {
			panic(err)
		}

		if !checkUint16(data) {
			t.Fail()
		}
	}
}

func TestReadUint8(t *testing.T) {

	files := getFlist("u1")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetUint8()
		if err != nil {
			panic(err)
		}

		if !checkUint8(data) {
			t.Fail()
		}
	}
}

func TestReadInt64(t *testing.T) {

	files := getFlist("i8")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetInt64()
		if err != nil {
			panic(err)
		}

		if !checkInt64(data) {
			t.Fail()
		}
	}
}

func TestReadInt32(t *testing.T) {

	files := getFlist("i4")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetInt32()
		if err != nil {
			panic(err)
		}

		if !checkInt32(data) {
			t.Fail()
		}
	}
}

func TestReadInt16(t *testing.T) {

	files := getFlist("i2")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetInt16()
		if err != nil {
			panic(err)
		}

		if !checkInt16(data) {
			t.Fail()
		}
	}
}

func TestReadInt8(t *testing.T) {

	files := getFlist("i1")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetInt8()
		if err != nil {
			panic(err)
		}

		if !checkInt8(data) {
			t.Fail()
		}
	}
}

func TestReadBytes(t *testing.T) {

	files := getFlist("i1")

	for _, fname := range files {

		fid, err := os.Open(path.Join("data", fname))
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(fid)
		if err != nil {
			panic(err)
		}
		data, err := rdr.GetBytes()
		if err != nil {
			panic(err)
		}

		if !checkBytes(data) {
			t.Fail()
		}
	}
}
