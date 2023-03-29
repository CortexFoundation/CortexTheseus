package banderwagon

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/crate-crypto/go-ipa/bandersnatch"
	"github.com/crate-crypto/go-ipa/bandersnatch/fr"
	"github.com/crate-crypto/go-ipa/common/parallel"
	"golang.org/x/sync/errgroup"
)

const (
	// optimized16BitIdxs is how many elements counting from the first element of the SRS we use a 16-bit table.
	// For the rest of the (numPoints-optimized16bitIdxs) elements we use a 8-bit table.
	optimized16BitIdxs = 5
)

// PrecomputeLagrange contains precomputed tables for a SRS.
type PrecomputeLagrange struct {
	// numPoints is the number of points in the SRS.
	numPoints int
	// inner16Bit contains the precomputed tables for the first `optimized16BitIdx` group elements.
	inner16Bit []*LagrangeTablePoints
	// inner8Bit contains the precomputed tables for the rest of the group elements.
	inner8Bit []*LagrangeTablePoints
}

// Equal returns true if the two PrecomputeLagrange are equal.
func (pcl PrecomputeLagrange) Equal(other PrecomputeLagrange) bool {
	if pcl.numPoints != other.numPoints {
		return false
	}
	if len(pcl.inner8Bit) != len(other.inner8Bit) {
		return false
	}
	if len(pcl.inner16Bit) != len(other.inner16Bit) {
		return false
	}
	for i := 0; i < len(pcl.inner8Bit); i++ {
		if !pcl.inner8Bit[i].Equal(*other.inner8Bit[i]) {
			return false
		}
	}
	for i := 0; i < len(pcl.inner16Bit); i++ {
		if !pcl.inner16Bit[i].Equal(*other.inner16Bit[i]) {
			return false
		}
	}
	return true
}

// NewPrecomputeLagrange creates a new PrecomputeLagrange from a set of points.
func NewPrecomputeLagrange(points []Element) *PrecomputeLagrange {
	pl := &PrecomputeLagrange{numPoints: len(points)}

	g, _ := errgroup.WithContext(context.Background())

	// Generate 16-bit table for points[:optimized16BitIdx]
	g.Go(func() error {
		numPoints := len(points)
		if numPoints > optimized16BitIdxs {
			numPoints = optimized16BitIdxs
		}
		table := make([]*LagrangeTablePoints, numPoints)
		parallel.Execute(numPoints, func(start, end int) {
			for i := start; i < end; i++ {
				// Each window have 1<<16 values, and we have a total of 256/16=16 windows.
				table[i] = newLagrangeTablePoints(points[i], 256/16, 1<<16)
			}
		})
		pl.inner16Bit = table
		return nil
	})

	// Generate the 8-bit table for points[optimized16BitIdx:]
	if len(points)-optimized16BitIdxs > 0 {
		g.Go(func() error {
			numPoints := len(points) - optimized16BitIdxs
			table := make([]*LagrangeTablePoints, numPoints)
			parallel.Execute(numPoints, func(start, end int) {
				// We generate the table, but just shifted `optimized16BitIdxs` positions,
				// since those group elements live in the 16-bit table.
				for i := start; i < end; i++ {
					// Each window have 1<<8 values, and we have a total of 256/8=32 windows.
					table[i] = newLagrangeTablePoints(points[i+optimized16BitIdxs], 256/8, 1<<8)
				}
			})
			pl.inner8Bit = table
			return nil
		})
	}
	g.Wait()

	return pl
}

// SerializePrecomputedLagrange serializes a PrecomputeLagrange.
// The format is:
// [int64(numPoints)][int64(8bitTableCount)][8BitTable1]...[8BitTableN][int64(16bitTableCount)][16BitTable1]...[16BitTableN]
// See (*LagrangeTablePoints).Serialize() for the format of the tables.
func (pcl *PrecomputeLagrange) SerializePrecomputedLagrange(w io.Writer) error {
	err := binary.Write(w, binary.LittleEndian, int64(pcl.numPoints))
	if err != nil {
		return fmt.Errorf("serializing the number of points: %s", err)
	}

	// Serialize 8-bit tables.
	if err := binary.Write(w, binary.LittleEndian, int64(len(pcl.inner8Bit))); err != nil {
		return fmt.Errorf("serializing the number of points for 8-bit table: %s", err)
	}
	for i := range pcl.inner8Bit {
		if err := pcl.inner8Bit[i].Serialize(w); err != nil {
			return fmt.Errorf("serializing 8-bit table for %d-th point: %s", i, err)
		}
	}

	// Serialize 16-bit tables.
	if err := binary.Write(w, binary.LittleEndian, int64(len(pcl.inner16Bit))); err != nil {
		return fmt.Errorf("serializing the number of points for 16-bit table: %s", err)
	}
	for i := range pcl.inner16Bit {
		if err := pcl.inner16Bit[i].Serialize(w); err != nil {
			return fmt.Errorf("serializing 16-bit table for %d-th point: %s", i, err)
		}
	}

	return nil
}

// DeserializePrecomputedLagrange deserializes a PrecomputeLagrange.
// See SerializePrecomputedLagrange() for the format description.
func DeserializePrecomputedLagrange(reader io.Reader) (*PrecomputeLagrange, error) {
	var pcl PrecomputeLagrange

	var numPoints int64
	if err := binary.Read(reader, binary.LittleEndian, &numPoints); err != nil {
		return nil, fmt.Errorf("deserializing the number of points: %s", err)
	}
	pcl.numPoints = int(numPoints)

	// 8-bit table deserialization.
	var table8BitCount int64
	if err := binary.Read(reader, binary.LittleEndian, &table8BitCount); err != nil {
		return nil, fmt.Errorf("deserializing the number of points for 8-bit table: %s", err)
	}
	pcl.inner8Bit = make([]*LagrangeTablePoints, table8BitCount)
	for i := 0; i < int(table8BitCount); i++ {
		pcl.inner8Bit[i] = &LagrangeTablePoints{}
		pcl.inner8Bit[i].Deserialize(reader)
	}

	// 16-bit table deserialization.
	var table16BitCount int64
	if err := binary.Read(reader, binary.LittleEndian, &table16BitCount); err != nil {
		return nil, fmt.Errorf("deserializing the number of points for 16-bit table: %s", err)
	}
	pcl.inner16Bit = make([]*LagrangeTablePoints, table16BitCount)
	for i := 0; i < int(table16BitCount); i++ {
		pcl.inner16Bit[i] = &LagrangeTablePoints{}
		pcl.inner16Bit[i].Deserialize(reader)
	}

	return &pcl, nil
}

// Commit computes the MSM of a set of evaluations.
func (p *PrecomputeLagrange) Commit(evaluations []fr.Element) Element {
	var result Element
	result.Identity()

	// We use p.inner16Bits for the first 5 group elements.
	for i := 0; i < len(evaluations) && i < len(p.inner16Bit); i++ {
		scalar := &evaluations[i]

		if scalar.IsZero() {
			continue
		}

		table := p.inner16Bit[i]
		scalar_bytes_le := scalar.BytesLE()

		for row := 0; row < 16; row++ {
			value := uint16(scalar_bytes_le[2*row]) + uint16(scalar_bytes_le[2*row+1])<<8
			if value == 0 {
				continue
			}
			tp := table.point(row, value)
			result.AddMixed(&result, *tp)
		}
	}

	// We use p.inner8Bits for the rest of the elements.
	for i := len(p.inner16Bit); i < len(evaluations); i++ {
		scalar := &evaluations[i]

		if scalar.IsZero() {
			continue
		}
		table := p.inner8Bit[i-len(p.inner16Bit)]
		scalar_bytes_le := scalar.BytesLE()

		for row, value := range scalar_bytes_le {
			if value == 0 {
				continue
			}
			tp := table.point(row, uint16(value))
			result.AddMixed(&result, *tp)
		}
	}

	return result
}

type LagrangeTablePoints struct {
	identity bandersnatch.PointAffine // TODO We can save memory by removing this
	// windowSize is the window size for each index.
	// e.g: point(index, value) = matrix[i *windowSize + value]
	windowSize int
	matrix     []bandersnatch.PointAffine
}

// Serialize serializes a LagrangeTablePoints in the following format:
// [int64(numRows)][int64(windowSize)][point1]...[pointN]
// Where [pointX] is an affine point in uncompressed form.
func (ltp *LagrangeTablePoints) Serialize(w io.Writer) error {
	// Number of rows.
	if err := binary.Write(w, binary.LittleEndian, int64(len(ltp.matrix))); err != nil {
		return fmt.Errorf("writing column count: %s", err)
	}
	// Window size.
	if err := binary.Write(w, binary.LittleEndian, int64(ltp.windowSize)); err != nil {
		return fmt.Errorf("writing window size: %s", err)
	}
	// Write points in affine uncompressed form.
	for _, p := range ltp.matrix {
		p.WriteUncompressedPoint(w)
	}

	return nil
}

// Deserialize deserializes a LagrangeTablePoints.
// See (*LagrangeTablePoints).Serialize() for the format description.
func (ltp *LagrangeTablePoints) Deserialize(r io.Reader) error {
	var columnCount int64
	if err := binary.Read(r, binary.LittleEndian, &columnCount); err != nil {
		return fmt.Errorf("deserializing the number of columns: %s", err)
	}
	var windowSize int64
	if err := binary.Read(r, binary.LittleEndian, &windowSize); err != nil {
		return fmt.Errorf("deserializing window size: %s", err)
	}
	ltp.identity.Identity()
	ltp.windowSize = int(windowSize)
	ltp.matrix = make([]bandersnatch.PointAffine, columnCount)
	for i := range ltp.matrix {
		ltp.matrix[i] = bandersnatch.ReadUncompressedPoint(r)
	}
	return nil
}

// Equal returns true if the two LagrangeTablePoints are equal.
func (ltp LagrangeTablePoints) Equal(other LagrangeTablePoints) bool {
	if len(ltp.matrix) != len(other.matrix) {
		return false
	}

	if ltp.identity != other.identity {
		return false
	}

	for i := 0; i < len(ltp.matrix); i++ {
		if !ltp.matrix[i].Equal(&other.matrix[i]) {
			return false
		}
	}
	return true
}

// NewLagrangTablePoints creates a new LagrangeTablePoints.
func NewLagrangeTablePoints(point Element, num_rows int, base_int int) *LagrangeTablePoints {
	return newLagrangeTablePoints(point, num_rows, base_int)
}

func newLagrangeTablePoints(point Element, num_rows int, base_int int) *LagrangeTablePoints {
	var base fr.Element
	base.SetUint64(uint64(base_int))

	base_row := compute_base_row(point, (base_int - 1))

	rows := make([]Element, 0, num_rows*(base_int-1))
	rows = append(rows, base_row...)

	scale := base
	// TODO: we can do this in parallel
	for i := 1; i < num_rows; i++ {
		scaled_row := scale_row(base_row, scale)
		rows = append(rows, scaled_row...)
		scale.Mul(&scale, &base)
	}
	rows_affine := elements_to_affine(rows)
	var identity bandersnatch.PointAffine
	identity.Identity()
	return &LagrangeTablePoints{
		identity:   identity,
		windowSize: base_int - 1, // Zero is not included.
		matrix:     rows_affine,
	}
}

func (ltp *LagrangeTablePoints) point(index int, value uint16) *bandersnatch.PointAffine {
	if value == 0 {
		return &ltp.identity
	}
	return &ltp.matrix[uint(index*ltp.windowSize)+uint(value-1)]
}

func compute_base_row(point Element, num_points int) []Element {
	row := make([]Element, num_points)
	row[0] = point

	for i := 1; i < num_points; i++ {
		var row_i Element
		row_i.Add(&row[i-1], &point)
		row[i] = row_i
	}
	return row
}

func scale_row(points []Element, scale fr.Element) []Element {
	scaled_points := make([]Element, len(points))
	for i := 0; i < len(points); i++ {

		scaled_points[i].ScalarMul(&points[i], &scale)
		scaled_points[i].Normalise()
	}
	return scaled_points
}

func elements_to_affine(points []Element) []bandersnatch.PointAffine {
	affine_points := make([]bandersnatch.PointAffine, len(points))

	for index, point := range points {
		var affine bandersnatch.PointAffine
		affine.FromProj(&point.inner)
		affine_points[index] = affine
	}

	return affine_points
}
