package ipa

import (
	"encoding/binary"
	"io"

	"github.com/crate-crypto/go-ipa/bandersnatch/fr"
	"github.com/crate-crypto/go-ipa/banderwagon"
	"github.com/crate-crypto/go-ipa/common"
)

type IPAProof struct {
	L        []banderwagon.Element
	R        []banderwagon.Element
	A_scalar fr.Element
}

func CreateIPAProof(transcript *common.Transcript, ic *IPAConfig, commitment banderwagon.Element, a []fr.Element, eval_point fr.Element) IPAProof {
	transcript.DomainSep("ipa")

	b := ic.PrecomputedWeights.ComputeBarycentricCoefficients(eval_point)
	inner_prod := InnerProd(a, b)

	transcript.AppendPoint(&commitment, "C")
	transcript.AppendScalar(&eval_point, "input point")
	transcript.AppendScalar(&inner_prod, "output point")
	w := transcript.ChallengeScalar("w")

	var q banderwagon.Element
	q.ScalarMul(&ic.SRSPrecompPoints.Q, &w)

	num_rounds := ic.num_ipa_rounds

	current_basis := ic.SRSPrecompPoints.SRS

	L := make([]banderwagon.Element, num_rounds)
	R := make([]banderwagon.Element, num_rounds)

	for i := 0; i < int(num_rounds); i++ {

		a_L, a_R := splitScalars(a)

		b_L, b_R := splitScalars(b)

		G_L, G_R := splitPoints(current_basis)

		z_L := InnerProd(a_R, b_L)
		z_R := InnerProd(a_L, b_R)

		C_L_1 := commit(G_L, a_R)
		C_L := commit([]banderwagon.Element{C_L_1, q}, []fr.Element{fr.One(), z_L})

		C_R_1 := commit(G_R, a_L)
		C_R := commit([]banderwagon.Element{C_R_1, q}, []fr.Element{fr.One(), z_R})

		L[i] = C_L
		R[i] = C_R

		transcript.AppendPoint(&C_L, "L")
		transcript.AppendPoint(&C_R, "R")
		x := transcript.ChallengeScalar("x")

		var xInv fr.Element
		xInv.Inverse(&x)

		// TODO: We could use a for loop here like in the Rust code
		a = foldScalars(a_L, a_R, x)
		b = foldScalars(b_L, b_R, xInv)

		current_basis = foldPoints(G_L, G_R, xInv)

	}

	if len(a) != 1 {
		panic("length of `a` should be 1 at the end of the reduction")
	}

	return IPAProof{
		L:        L,
		R:        R,
		A_scalar: a[0],
	}
}

func (ip *IPAProof) Write(w io.Writer) {
	for _, el := range ip.L {
		binary.Write(w, binary.BigEndian, el.Bytes())
	}
	for _, ar := range ip.R {
		binary.Write(w, binary.BigEndian, ar.Bytes())
	}
	binary.Write(w, binary.BigEndian, ip.A_scalar.BytesLE())
}

func (ip *IPAProof) Read(r io.Reader) {
	var L []banderwagon.Element
	for i := 0; i < 8; i++ {
		L_i := common.ReadPoint(r)
		L = append(L, *L_i)
	}
	ip.L = L
	var R []banderwagon.Element
	for i := 0; i < 8; i++ {
		R_i := common.ReadPoint(r)
		R = append(R, *R_i)
	}
	ip.R = R

	A_Scalar := common.ReadScalar(r)
	ip.A_scalar = *A_Scalar
}

func (ip IPAProof) Equal(other IPAProof) bool {
	num_rounds := 8
	if len(ip.L) != len(other.L) {
		return false
	}
	if len(ip.R) != len(other.R) {
		return false
	}
	if len(ip.L) != len(ip.R) {
		return false
	}
	if len(ip.L) != num_rounds {
		return false
	}

	for i := 0; i < num_rounds; i++ {
		expect_L_i := ip.L[i]
		expect_R_i := ip.R[i]

		got_L_i := other.L[i]
		got_R_i := other.R[i]

		if !expect_L_i.Equal(&got_L_i) {
			return false
		}
		if !expect_R_i.Equal(&got_R_i) {
			return false
		}
	}
	return ip.A_scalar.Equal(&other.A_scalar)
}
