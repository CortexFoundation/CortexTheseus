package common

import (
	"io"

	"github.com/crate-crypto/go-ipa/bandersnatch/fr"
	"github.com/crate-crypto/go-ipa/banderwagon"
)

// TODO: This is not entirely correct, the degree is 255. We can change this to VECTOR_LENGTH or NUM_EVAL_POINTS?
// note: degree 255, means 256 evaluation points
const POLY_DEGREE = 256

// Returns powers of x from 0 to degree-1
// <1, x, x^2, x^3, x^4,...,x^(degree-1)>
// TODO This method is used in two places; one is to evaluate a polynomial (test), and the other is to
// TODO compute powers of challenges.
// TODO the first one we can use the bls package for
// TODO The second we _could_ just multiply on each iteration, (depends on how readable it is)
func PowersOf(x fr.Element, degree int) []fr.Element {
	result := make([]fr.Element, degree)
	result[0] = fr.One()

	for i := 1; i < degree; i++ {
		result[i].Mul(&result[i-1], &x)
	}

	return result
}

func ReadPoint(r io.Reader) *banderwagon.Element {
	var x = make([]byte, 32)
	n, err := r.Read(x)
	if err != nil {
		panic("error reading bytes")
	}
	if n != 32 {
		panic("did not read enough bytes")
	}
	var p = &banderwagon.Element{}
	err = p.SetBytes(x)
	if err != nil {
		panic("could not deserialize point")
	}
	return p
}

func ReadScalar(r io.Reader) *fr.Element {
	var x = make([]byte, 32)
	n, err := r.Read(x)
	if err != nil {
		panic("error reading bytes")
	}
	if n != 32 {
		panic("did not read enough bytes")
	}
	var scalar = &fr.Element{}
	scalar.SetBytesLE(x)
	if err != nil {
		panic("could not deserialize point")
	}
	return scalar
}
