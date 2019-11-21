package inference

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// NpyWriter can write data from a Go slice to a Numpy binary array.
type NpyWriter struct {

	// Defaults to little endian, but can be set to
	// binary.BigEndian before writing data
	Endian binary.ByteOrder

	// Defaults to nx1, where n is the length of data.  Can be set
	// to any shape with any number of dimensions.  The shape is
	// not checked for compatibility with the data.
	Shape []int

	// Defaults to false (row major order), can be set to true
	// (column major order) before writing the data.
	ColumnMajor bool

	// Defaults to 1, can be set to 2 before writing the data.
	Version int

	// The writer to which the data are written
	w io.WriteCloser
}

// NewFileWriter returns a NpyWriter that can be used to write data to
// a Numpy binary format file.  After creation, call one of the
// WriteXX methods to write array data to the file.  The file is
// automatically closed at the end of that call.  Only one array can
// be written to a file.
func NewFileWriter(fname string) (*NpyWriter, error) {

	w, err := os.Create(fname)
	if err != nil {
		return nil, err
	}

	return NewWriter(w)
}

// NewWriter returns a NpyWriter that can be used to write data to an
// io.WriteCloser, using the Numpy binary format.  After creation,
// call one of the WriteXXX methods to write array data to the writer.
// The file is automatically closed at the end of that call.  Only one
// slice can be written to the writer.
func NewWriter(w io.WriteCloser) (*NpyWriter, error) {

	wtr := &NpyWriter{
		w:       w,
		Endian:  binary.LittleEndian,
		Version: 1,
	}

	return wtr, nil
}

func (wtr *NpyWriter) writeHeader(dtype string, length int) error {

	_, err := wtr.w.Write([]byte("\x93NUMPY"))
	if err != nil {
		return err
	}

	err = binary.Write(wtr.w, binary.LittleEndian, uint8(wtr.Version))
	if err != nil {
		return err
	}

	err = binary.Write(wtr.w, binary.LittleEndian, uint8(0))
	if err != nil {
		return err
	}

	if wtr.Endian == binary.LittleEndian {
		dtype = "<" + dtype
	} else {
		dtype = ">" + dtype
	}

	var shapeString string
	if wtr.Shape != nil {
		shapeString = ""
		for _, v := range wtr.Shape {
			shapeString += fmt.Sprintf("%d,", v)
		}
		shapeString = "(" + shapeString + ")"
	} else {
		shapeString = fmt.Sprintf("(%d,)", length)
	}

	cmaj := "False"
	if wtr.ColumnMajor {
		cmaj = "True"
	}

	header := fmt.Sprintf("{'descr': '%s', 'fortran_order': %s, 'shape': %s,}",
		dtype, cmaj, shapeString)

	pad := 16 - ((10 + len(header)) % 16)
	if wtr.Version == 2 {
		pad = 32 - ((10 + len(header)) % 32)
	}
	if pad > 0 {
		header += strings.Repeat(" ", pad)
	}

	if wtr.Version == 1 {
		err = binary.Write(wtr.w, binary.LittleEndian, uint16(len(header)))
	} else {
		err = binary.Write(wtr.w, binary.LittleEndian, uint32(len(header)))
	}
	if err != nil {
		return err
	}

	_, err = wtr.w.Write([]byte(header))
	if err != nil {
		return err
	}

	return nil
}
