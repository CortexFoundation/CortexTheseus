package utils

import (
	"github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
)

// The spec includes a method to compute the modular inverse.
// This method is named .Inverse on `fr.Element`
// When the element to invert is zero, this method will return zero
// however note that this is not utilized in the specs anywhere
// and so it is also fine to panic on zero.
//
// [bls_modular_inverse]: https://github.com/ethereum/consensus-specs/blob/50a3f8e8d902ad9d677ca006302eb9535d56d758/specs/deneb/polynomial-commitments.md#bls_modular_inverse
// [div]: https://github.com/ethereum/consensus-specs/blob/50a3f8e8d902ad9d677ca006302eb9535d56d758/specs/deneb/polynomial-commitments.md#div

// ComputePowers computes x^0 to x^n-1.
//
// More precisely, given x and n, returns a slice containing [x^0, ..., x^n-1]
// In particular, for n==0, an empty slice is returned
//
// [compute_powers]: https://github.com/ethereum/consensus-specs/blob/50a3f8e8d902ad9d677ca006302eb9535d56d758/specs/deneb/polynomial-commitments.md#compute_powers
func ComputePowers(x fr.Element, n uint) []fr.Element {
	if n == 0 {
		return []fr.Element{}
	}

	powers := make([]fr.Element, n)
	powers[0].SetOne()
	for i := uint(1); i < n; i++ {
		powers[i].Mul(&powers[i-1], &x)
	}

	return powers
}

// IsPowerOfTwo returns true if `value` is a power of two.
//
// `0` will return false
//
// [is_power_of_two]: https://github.com/ethereum/consensus-specs/blob/50a3f8e8d902ad9d677ca006302eb9535d56d758/specs/deneb/polynomial-commitments.md#is_power_of_two
func IsPowerOfTwo(value uint64) bool {
	return value > 0 && (value&(value-1) == 0)
}

// Reverse reverses the list in-place
func Reverse[K interface{}](list []K) {
	lastIndex := len(list) - 1
	for i := 0; i < len(list)/2; i++ {
		list[i], list[lastIndex-i] = list[lastIndex-i], list[i]
	}
}

// Tries to convert a byte slice to a field element.
// Returns an error if the byte slice was not a canonical representation
// of the field element.
// Canonical meaning that the big integer interpretation was less than
// the field's prime. ie it lies within the range [0, p-1] (inclusive)
func ReduceCanonicalLittleEndian(serScalar []byte) (fr.Element, error) {
	// gnark uses big-endian but the format is in little endian
	// so we reverse the bytes
	Reverse(serScalar[:])
	return reduceCanonicalBigEndian(serScalar)
}

func reduceCanonicalBigEndian(serScalar []byte) (fr.Element, error) {
	var scalar fr.Element
	err := scalar.SetBytesCanonical(serScalar)

	return scalar, err
}
