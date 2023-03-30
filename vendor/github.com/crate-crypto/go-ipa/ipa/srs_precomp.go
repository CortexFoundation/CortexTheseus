package ipa

import (
	"bytes"
	"encoding/binary"

	"github.com/crate-crypto/go-ipa/banderwagon"
)

// Stores the SRS and the precomputed SRS points too
type SRSPrecompPoints struct {
	// Points to commit to the input vector
	// We could try to find these points in the precomputed points
	// but for now, we just store the SRS too
	SRS []banderwagon.Element
	// Point to commit to the inner product of the two vectors in the inner product argument
	Q banderwagon.Element
	// Precomputed SRS points
	PrecompLag *banderwagon.PrecomputeLagrange
}

// NewSRSPrecomp returns an instance a SRS with the given number of points, and generates
// a precomputed table for them.
func NewSRSPrecomp(num_points uint) *SRSPrecompPoints {
	srs := GenerateRandomPoints(uint64(num_points))
	var Q banderwagon.Element = banderwagon.Generator
	preComp := banderwagon.NewPrecomputeLagrange(srs)

	return &SRSPrecompPoints{
		SRS:        srs,
		Q:          Q,
		PrecompLag: preComp,
	}
}

// SerializeSRSPrecomp serializes the precomputed table into a byte slice.
// The format is: [in64(len(SRS))] [SRS points (uncompressed)] [Precomp table]
// To see the format of [Precomp table], refer to (*PrecomputeLagrange).SerializePrecomputedLagrange().
func (spc *SRSPrecompPoints) SerializeSRSPrecomp() ([]byte, error) {
	var buf bytes.Buffer

	err := binary.Write(&buf, binary.LittleEndian, int64(len(spc.SRS)))
	if err != nil {
		return nil, err
	}

	for _, p := range spc.SRS {
		p.UnsafeWriteUncompressedPoint(&buf)
	}

	err = spc.PrecompLag.SerializePrecomputedLagrange(&buf)
	if err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func DeserializeSRSPrecomp(serialized []byte) (*SRSPrecompPoints, error) {
	var spc SRSPrecompPoints
	reader := bytes.NewReader(serialized)

	var lenSRS int64
	err := binary.Read(reader, binary.LittleEndian, &lenSRS)
	if err != nil {
		return nil, err
	}
	spc.SRS = make([]banderwagon.Element, lenSRS)

	for i := 0; i < int(lenSRS); i++ {
		spc.SRS[i] = *banderwagon.UnsafeReadUncompressedPoint(reader)
	}

	pcl, err := banderwagon.DeserializePrecomputedLagrange(reader)
	if err != nil {
		return nil, err
	}
	spc.PrecompLag = pcl
	spc.Q = banderwagon.Generator

	return &spc, nil
}

func (spc SRSPrecompPoints) Equal(other SRSPrecompPoints) bool {
	if len(spc.SRS) != len(other.SRS) {
		return false
	}

	for i := 0; i < len(spc.SRS); i++ {
		if !spc.SRS[i].Equal(&other.SRS[i]) {
			return false
		}
	}

	if !spc.Q.Equal(&other.Q) {
		return false
	}

	return spc.PrecompLag.Equal(*other.PrecompLag)
}
