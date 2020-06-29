// GENERATED CODE, DO NOT EDIT

package inference

import (
	"os"
	"testing"
)

func TestWriteComplex128(t *testing.T) {

	data := []complex128{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteComplex128(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetComplex128()
		if err != nil {
			panic(err)
		}

		if !checkComplex128(data) {
			t.Fail()
		}
	}
}

func TestWriteComplex64(t *testing.T) {

	data := []complex64{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteComplex64(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetComplex64()
		if err != nil {
			panic(err)
		}

		if !checkComplex64(data) {
			t.Fail()
		}
	}
}

func TestWriteFloat64(t *testing.T) {

	data := []float64{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteFloat64(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetFloat64()
		if err != nil {
			panic(err)
		}

		if !checkFloat64(data) {
			t.Fail()
		}
	}
}

func TestWriteFloat32(t *testing.T) {

	data := []float32{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteFloat32(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetFloat32()
		if err != nil {
			panic(err)
		}

		if !checkFloat32(data) {
			t.Fail()
		}
	}
}

func TestWriteUint64(t *testing.T) {

	data := []uint64{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteUint64(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetUint64()
		if err != nil {
			panic(err)
		}

		if !checkUint64(data) {
			t.Fail()
		}
	}
}

func TestWriteUint32(t *testing.T) {

	data := []uint32{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteUint32(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetUint32()
		if err != nil {
			panic(err)
		}

		if !checkUint32(data) {
			t.Fail()
		}
	}
}

func TestWriteUint16(t *testing.T) {

	data := []uint16{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteUint16(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetUint16()
		if err != nil {
			panic(err)
		}

		if !checkUint16(data) {
			t.Fail()
		}
	}
}

func TestWriteUint8(t *testing.T) {

	data := []uint8{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteUint8(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetUint8()
		if err != nil {
			panic(err)
		}

		if !checkUint8(data) {
			t.Fail()
		}
	}
}

func TestWriteInt64(t *testing.T) {

	data := []int64{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteInt64(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetInt64()
		if err != nil {
			panic(err)
		}

		if !checkInt64(data) {
			t.Fail()
		}
	}
}

func TestWriteInt32(t *testing.T) {

	data := []int32{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteInt32(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetInt32()
		if err != nil {
			panic(err)
		}

		if !checkInt32(data) {
			t.Fail()
		}
	}
}

func TestWriteInt16(t *testing.T) {

	data := []int16{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteInt16(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetInt16()
		if err != nil {
			panic(err)
		}

		if !checkInt16(data) {
			t.Fail()
		}
	}
}

func TestWriteInt8(t *testing.T) {

	data := []int8{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteInt8(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetInt8()
		if err != nil {
			panic(err)
		}

		if !checkInt8(data) {
			t.Fail()
		}
	}
}

func TestWriteBytes(t *testing.T) {

	data := []byte{0, 1, 2, 3, 4, 5, 6, 7}

	for shape := 0; shape < 4; shape++ {

		wtr, err := NewFileWriter("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		switch shape {
		case 0:
			wtr.Shape = []int{4, 2}
		case 1:
			wtr.Shape = []int{8, 1}
		case 2:
			wtr.Shape = []int{1, 8}
		case 3:
			wtr.Shape = nil
		}

		err = wtr.WriteBytes(data)
		if err != nil {
			panic(err)
		}

		r, err := os.Open("data/tmp.npy")
		if err != nil {
			panic(err)
		}

		rdr, err := NewReader(r)
		if err != nil {
			panic(err)
		}

		data, err = rdr.GetBytes()
		if err != nil {
			panic(err)
		}

		if !checkBytes(data) {
			t.Fail()
		}
	}
}
