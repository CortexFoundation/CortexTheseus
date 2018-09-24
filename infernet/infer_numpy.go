package infernet

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// NpyReader can read data from a Numpy binary array into a Go slice.
type NpyReader struct {

	// The numpy data type of the array
	Dtype string

	// The endianness of the binary data
	Endian binary.ByteOrder

	// The version number of the file format
	Version int

	// The shape of the array as specified in the file.
	Shape []int

	// If true, the data are flattened in column-major order,
	// otherwise they are flattened in row-major order.
	ColumnMajor bool

	// Read the data from this source
	r io.Reader

	// Number of elements in the array to be read (obtained from
	// header).
	n_elt int
}

// NewFileReader is a convenience method returning a NpyReader that
// can be used to obtain array data from the given named file.  Call
// one of the GetXXX methods to obtain the data slice.
func NewFileReader(f string) (*NpyReader, error) {

	fid, err := os.Open(f)
	if err != nil {
		return nil, err
	}
	r, err := NewReader(fid)
	return r, err
}

// Parse the shape string in the file header.
func parseShape(header []byte) ([]int, int, error) {

	re := regexp.MustCompile(`'shape':\s*\(([^\(]*)\)`)
	ma := re.FindSubmatch(header)
	if ma == nil {
		return nil, 0, fmt.Errorf("Shape not found in header.\n")
	}

	shapes := string(ma[1])
	shape := make([]int, 0)
	n_elt := 1
	for _, s := range strings.Split(shapes, ",") {
		s = strings.Trim(s, " ")
		if len(s) == 0 {
			break
		}
		x, err := strconv.Atoi(s)
		if err != nil {
			panic(err)
		}
		n_elt *= x
		shape = append(shape, x)
	}

	return shape, n_elt, nil
}

// NewReader returns a NpyReader that can be used to obtain array data
// as a Go slice.  The Go slice has a type matching the dtype in the
// Numpy file.  Call one of the GetXX methods to obtain the slice.
func NewReader(r io.Reader) (*NpyReader, error) {

	// Check the magic number
	b := make([]byte, 6)
	n, err := r.Read(b)
	if err != nil {
		return nil, err
	} else if n != 6 {
		return nil, fmt.Errorf("Input appears to be truncated")
	} else if string(b) != "\x93NUMPY" {
		return nil, fmt.Errorf("Not npy format data (wrong magic number)")
	}

	// Get the major version number
	var version uint8
	err = binary.Read(r, binary.LittleEndian, &version)
	if err != nil {
		return nil, err
	}
	if version != 1 && version != 2 {
		return nil, fmt.Errorf("Invalid version number %d", version)
	}

	// Check the minor version number
	var minor uint8
	err = binary.Read(r, binary.LittleEndian, &minor)
	if err != nil {
		return nil, err
	}
	if minor != 0 {
		return nil, fmt.Errorf("Invalid minor version number %d", version)
	}

	// Get the size in bytes of the header
	var header_length int
	if version == 1 {
		var hl uint16
		err = binary.Read(r, binary.LittleEndian, &hl)
		header_length = int(hl)
	} else {
		var hl uint32
		err = binary.Read(r, binary.LittleEndian, &hl)
		header_length = int(hl)
	}
	if err != nil {
		return nil, err
	}

	// Read the header
	header := make([]byte, header_length)
	_, err = r.Read(header)
	if err != nil {
		return nil, err
	}

	// Get the dtype
	re := regexp.MustCompile(`'descr':\s*'([^']*)'`)
	ma := re.FindSubmatch(header)
	if ma == nil {
		return nil, fmt.Errorf("dtype description not found in header")
	}
	dtype := string(ma[1])

	// Get the order information
	re = regexp.MustCompile(`'fortran_order':\s*(False|True)`)
	ma = re.FindSubmatch(header)
	if ma == nil {
		return nil, fmt.Errorf("fortran_order not found in header")
	}
	fortran_order := string(ma[1])

	// Get the shape information
	shape, n_elt, err := parseShape(header)
	if err != nil {
		return nil, err
	}

	var endian binary.ByteOrder
	if strings.HasPrefix(dtype, ">") {
		endian = binary.BigEndian
	} else {
		// Default
		endian = binary.LittleEndian
	}

	rdr := &NpyReader{
		Dtype:       dtype[1:],
		ColumnMajor: fortran_order == "True",
		Shape:       shape,
		Endian:      endian,
		Version:     int(version),
		n_elt:       n_elt,
		r:           r,
	}

	return rdr, nil
}

// GetFloat64 returns the array data as a slice of float64 values.
func (rdr *NpyReader) GetFloat64() ([]float64, error) {

	if rdr.Dtype != "f8" {
		return nil, fmt.Errorf("Reader does not contain float64 data")
	}

	data := make([]float64, rdr.n_elt)
	for k := 0; k < rdr.n_elt; k++ {
		err := binary.Read(rdr.r, rdr.Endian, &data[k])
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

// GetFloat32 returns the array data as a slice of float32 values.
func (rdr *NpyReader) GetFloat32() ([]float32, error) {

	if rdr.Dtype != "f4" {
		return nil, fmt.Errorf("Reader does not contain float32 data")
	}

	data := make([]float32, rdr.n_elt)
	for k := 0; k < rdr.n_elt; k++ {
		err := binary.Read(rdr.r, rdr.Endian, &data[k])
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

func (rdr *NpyReader) GetBytes() ([]byte, error) {

	data := make([]byte, rdr.n_elt)
	for k := 0; k < rdr.n_elt; k++ {
		err := binary.Read(rdr.r, rdr.Endian, &data[k])
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

func (rdr *NpyReader) GetUInt8() ([]uint8, error) {

	if rdr.Dtype != "u1" {
		return nil, fmt.Errorf("Reader does not contain uint8 data")
	}

	data := make([]uint8, rdr.n_elt)
	for k := 0; k < rdr.n_elt; k++ {
		err := binary.Read(rdr.r, rdr.Endian, &data[k])
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

// GetInt8 returns the array data as a slice of int8 values.
func (rdr *NpyReader) GetInt8() ([]int8, error) {

	if rdr.Dtype != "i1" {
		return nil, fmt.Errorf("Reader does not contain int8 data")
	}

	data := make([]int8, rdr.n_elt)
	for k := 0; k < rdr.n_elt; k++ {
		err := binary.Read(rdr.r, rdr.Endian, &data[k])
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

// GetInt16 returns the array data as a slice of int16 values.
func (rdr *NpyReader) GetInt16() ([]int16, error) {

	if rdr.Dtype != "i2" {
		return nil, fmt.Errorf("Reader does not contain int16 data")
	}

	data := make([]int16, rdr.n_elt)
	for k := 0; k < rdr.n_elt; k++ {
		err := binary.Read(rdr.r, rdr.Endian, &data[k])
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

// GetInt32 returns the array data as a slice of int32 values.
func (rdr *NpyReader) GetInt32() ([]int32, error) {

	if rdr.Dtype != "i4" {
		return nil, fmt.Errorf("Reader does not contain int32 data")
	}

	data := make([]int32, rdr.n_elt)
	for k := 0; k < rdr.n_elt; k++ {
		err := binary.Read(rdr.r, rdr.Endian, &data[k])
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

// GetInt64 returns the array data as a slice of int64 values.
func (rdr *NpyReader) GetInt64() ([]int64, error) {

	if rdr.Dtype != "i8" {
		return nil, fmt.Errorf("Reader does not contain int64 data")
	}

	data := make([]int64, rdr.n_elt)
	for k := 0; k < rdr.n_elt; k++ {
		err := binary.Read(rdr.r, rdr.Endian, &data[k])
		if err != nil {
			return nil, err
		}
	}

	return data, nil
}

func DumpToFile(fname string, data []byte) error {
	w, err := os.Create(fname)
	if err != nil {
		return err
	}

	for _, v := range data {
		err := binary.Write(w, binary.LittleEndian, v)
		if err != nil {
			return err
		}
	}

	w.Close()

	return nil
}
