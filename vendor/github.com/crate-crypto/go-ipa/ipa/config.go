package ipa

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"runtime"

	"github.com/crate-crypto/go-ipa/bandersnatch/fp"
	"github.com/crate-crypto/go-ipa/bandersnatch/fr"
	"github.com/crate-crypto/go-ipa/banderwagon"
	"github.com/crate-crypto/go-ipa/common"
)

type IPAConfig struct {
	// SRSPrecompPoints contains precomputed values for the SRS.
	SRSPrecompPoints *SRSPrecompPoints

	PrecomputedWeights *PrecomputedWeights

	// The number of rounds the prover and verifier must complete
	// in the IPA argument, this will be log2 of the size of the input vectors
	// since the vector is halved on each round
	num_ipa_rounds uint32
}

// This function creates common.POLY_DEGREE random generator points where the relative discrete log is
// not known between each generator and all of the other necessary information needed to verify
// and create an IPA proof.
func NewIPASettings() *IPAConfig {
	return &IPAConfig{
		SRSPrecompPoints:   NewSRSPrecomp(common.POLY_DEGREE),
		PrecomputedWeights: NewPrecomputedWeights(),
		num_ipa_rounds:     compute_num_rounds(common.POLY_DEGREE),
	}
}

func NewIPASettingsWithSRSPrecomp(srs_precomp *SRSPrecompPoints) *IPAConfig {
	return &IPAConfig{
		SRSPrecompPoints:   srs_precomp,
		PrecomputedWeights: NewPrecomputedWeights(),
		num_ipa_rounds:     compute_num_rounds(common.POLY_DEGREE),
	}
}

func MultiScalar(points []banderwagon.Element, scalars []fr.Element) banderwagon.Element {
	var result banderwagon.Element
	result.Identity()

	res, err := result.MultiExp(points, scalars, banderwagon.MultiExpConfig{NbTasks: runtime.NumCPU(), ScalarsMont: true})
	if err != nil {
		panic("mult exponentiation was not successful. TODO: replace panics by bubbling up error")
	}

	return *res
}

// Commits to a polynomial using the SRS
// panics if the length of the SRS does not equal the number of polynomial coefficients
func (ic *IPAConfig) Commit(polynomial []fr.Element) banderwagon.Element {
	return ic.SRSPrecompPoints.PrecompLag.Commit(polynomial)
}

// Commits to a polynomial using the input group elements
// panics if the number of group elements does not equal the number of polynomial coefficients
// This is used when the generators are not fixed
func commit(group_elements []banderwagon.Element, polynomial []fr.Element) banderwagon.Element {
	if len(group_elements) != len(polynomial) {
		panic(fmt.Sprintf("diff sizes, %d != %d", len(group_elements), len(polynomial)))
	}
	return MultiScalar(group_elements, polynomial)
}

// Computes the inner product of a and b
// panics if len(a) != len(b)
func InnerProd(a []fr.Element, b []fr.Element) fr.Element {
	if len(a) != len(b) {
		panic("two vectors must have the same lengths")
	}

	result := fr.Zero()
	for i := 0; i < len(a); i++ {
		var tmp fr.Element

		tmp.Mul(&a[i], &b[i])
		result.Add(&result, &tmp)
	}

	return result
}

// Computes c[i] =a[i] + b[i] * x
// returns c
// panics if len(a) != len(b)
func foldScalars(a []fr.Element, b []fr.Element, x fr.Element) []fr.Element {
	if len(a) != len(b) {
		panic("slices not equal length")
	}

	result := make([]fr.Element, len(a))
	for i := 0; i < len(a); i++ {
		var bx fr.Element
		bx.Mul(&x, &b[i])
		result[i].Add(&bx, &a[i])
	}
	return result
}

// Computes c[i] =a[i] + b[i] * x
// returns c
// panics if len(a) != len(b)
func foldPoints(a []banderwagon.Element, b []banderwagon.Element, x fr.Element) []banderwagon.Element {
	if len(a) != len(b) {
		panic("slices not equal length")
	}

	result := make([]banderwagon.Element, len(a))
	for i := 0; i < len(a); i++ {
		var bx banderwagon.Element
		bx.ScalarMul(&b[i], &x)
		result[i].Add(&bx, &a[i])
	}
	return result
}

// Splits a slice of scalars into two slices of equal length
// Eg [S1,S2,S3,S4] becomes [S1,S2] , [S3,S4]
// panics if the number of scalars is not even
func splitScalars(x []fr.Element) ([]fr.Element, []fr.Element) {
	if len(x)%2 != 0 {
		panic("slice should have an even length")
	}

	mid := len(x) / 2
	return x[:mid], x[mid:]
}

// Splits a slice of points into two slices of equal length
// Eg [P1,P2,P3,P4,P5,P6] becomes [P1,P2,P3] , [P4,P5,P6]
// panics if the number of points is not even
func splitPoints(x []banderwagon.Element) ([]banderwagon.Element, []banderwagon.Element) {
	if len(x)%2 != 0 {
		panic("slice should have an even length")
	}

	mid := len(x) / 2
	return x[:mid], x[mid:]
}

// This function does log2(vector_size)
//
// Since we do not allow for 0 size vectors, this is checked
// since we also do not allow for vectors which are not powers of 2, this is also checked
//
// It is okay to panic here, because the input is a constant, so it will panic before
// any proofs are made.
func compute_num_rounds(vector_size uint32) uint32 {
	// Check if this number is 0
	// zero is not a valid input to this function for our usecase
	if vector_size == 0 {
		panic("zero is not a valid input")
	}

	// See: https://stackoverflow.com/a/600306
	isPow2 := (vector_size & (vector_size - 1)) == 0

	if !isPow2 {
		panic("non power of 2 numbers are not valid inputs")
	}

	res := math.Log2(float64(vector_size))

	return uint32(res)
}

func GenerateRandomPoints(numPoints uint64) []banderwagon.Element {
	seed := "eth_verkle_oct_2021" // incase it changes or needs updating, we can use eth_verkle_month_year

	points := []banderwagon.Element{}

	var increment uint64 = 0

	for uint64(len(points)) != numPoints {

		digest := sha256.New()
		digest.Write([]byte(seed))

		b := make([]byte, 8)
		binary.BigEndian.PutUint64(b, increment)
		digest.Write(b)

		hash := digest.Sum(nil)

		var x fp.Element
		x.SetBytes(hash)

		increment++

		x_as_bytes := x.Bytes()
		var point_found banderwagon.Element
		err := point_found.SetBytes(x_as_bytes[:])
		if err != nil {
			// This point is not in the correct subgroup or on the curve
			continue
		}
		points = append(points, point_found)

	}

	return points
}
