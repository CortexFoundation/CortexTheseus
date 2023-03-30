package ipa

import (
	"github.com/crate-crypto/go-ipa/bandersnatch/fr"
	"github.com/crate-crypto/go-ipa/banderwagon"
	"github.com/crate-crypto/go-ipa/common"
)

func CheckIPAProof(transcript *common.Transcript, ic *IPAConfig, commitment banderwagon.Element, proof IPAProof, eval_point fr.Element, inner_prod fr.Element) bool {
	transcript.DomainSep("ipa")

	if len(proof.L) != len(proof.R) {
		panic("L and R should be the same size")
	}
	if len(proof.L) != int(ic.num_ipa_rounds) {
		panic("The number of points for L or R should be equal to the number of rounds")
	}

	b := ic.PrecomputedWeights.ComputeBarycentricCoefficients(eval_point)

	transcript.AppendPoint(&commitment, "C")
	transcript.AppendScalar(&eval_point, "input point")
	transcript.AppendScalar(&inner_prod, "output point")

	w := transcript.ChallengeScalar("w")

	var q banderwagon.Element
	q.ScalarMul(&ic.SRSPrecompPoints.Q, &w)

	var qy banderwagon.Element
	qy.ScalarMul(&q, &inner_prod)
	commitment.Add(&commitment, &qy)

	challenges := generateChallenges(transcript, &proof)
	challenges_inv := make([]fr.Element, len(challenges))

	// Compute expected commitment
	for i := 0; i < len(challenges); i++ {
		x := challenges[i]
		L := proof.L[i]
		R := proof.R[i]

		var xInv fr.Element
		xInv.Inverse(&x)

		challenges_inv[i] = xInv

		commitment = commit([]banderwagon.Element{commitment, L, R}, []fr.Element{fr.One(), x, xInv})
	}

	current_basis := ic.SRSPrecompPoints.SRS

	for i := 0; i < len(challenges); i++ {

		G_L, G_R := splitPoints(current_basis)

		b_L, b_R := splitScalars(b)

		xInv := challenges_inv[i]

		b = foldScalars(b_L, b_R, xInv)
		current_basis = foldPoints(G_L, G_R, xInv)
	}

	if len(b) != len(current_basis) {
		panic("reduction was not correct")
	}

	if len(b) != 1 {
		panic("`b` and `G` should be 1")
	}

	b0 := b[0]

	var got banderwagon.Element
	//  G[0] * a + (a * b) * Q;
	var part_1 banderwagon.Element
	part_1.ScalarMul(&current_basis[0], &proof.A_scalar)

	var part_2 banderwagon.Element
	var part_2a fr.Element

	part_2a.Mul(&b0, &proof.A_scalar)
	part_2.ScalarMul(&q, &part_2a)

	got.Add(&part_1, &part_2)

	return got.Equal(&commitment)
}

func generateChallenges(transcript *common.Transcript, proof *IPAProof) []fr.Element {

	challenges := make([]fr.Element, len(proof.L))
	for i := 0; i < len(proof.L); i++ {
		transcript.AppendPoint(&proof.L[i], "L")
		transcript.AppendPoint(&proof.R[i], "R")
		challenges[i] = transcript.ChallengeScalar("x")
	}
	return challenges
}
