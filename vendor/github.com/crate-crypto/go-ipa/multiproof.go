package multiproof

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/crate-crypto/go-ipa/bandersnatch/fr"
	"github.com/crate-crypto/go-ipa/banderwagon"
	"github.com/crate-crypto/go-ipa/common"
	"github.com/crate-crypto/go-ipa/ipa"
)

type MultiProof struct {
	IPA ipa.IPAProof
	D   banderwagon.Element
}

func CreateMultiProof(transcript *common.Transcript, ipaConf *ipa.IPAConfig, Cs []*banderwagon.Element, fs [][]fr.Element, zs []uint8) *MultiProof {
	transcript.DomainSep("multiproof")

	if len(Cs) != len(fs) {
		panic(fmt.Sprintf("number of commitments = %d, while number of functions = %d", len(Cs), len(fs)))
	}
	if len(Cs) != len(zs) {
		panic(fmt.Sprintf("number of commitments = %d, while number of points = %d", len(Cs), len(zs)))
	}

	num_queries := len(Cs)
	if num_queries == 0 {
		// TODO does this need to be a panic? no
		panic("cannot create a multiproof with 0 queries")
	}

	banderwagon.BatchNormalize(Cs)
	for i := 0; i < num_queries; i++ {
		transcript.AppendPoint(Cs[i], "C")
		var z = domainToFr(zs[i])
		transcript.AppendScalar(&z, "z")

		// get the `y` value

		f := fs[i]
		y := f[zs[i]]
		transcript.AppendScalar(&y, "y")
	}
	r := transcript.ChallengeScalar("r")
	powers_of_r := common.PowersOf(r, num_queries)

	// Compute g(x)
	// We first compute the polynomials in lagrange form grouped by evaluation point, and
	// then we compute g(X). This limit the numbers of DivideOnDomain() calls up to
	// the domain size.
	groupedFs := make([][]fr.Element, common.POLY_DEGREE)
	for i := 0; i < num_queries; i++ {
		z := zs[i]
		if len(groupedFs[z]) == 0 {
			groupedFs[z] = make([]fr.Element, common.POLY_DEGREE)
		}

		r := powers_of_r[i]
		for j := 0; j < common.POLY_DEGREE; j++ {
			var scaledEvaluation fr.Element
			scaledEvaluation.Mul(&r, &fs[i][j])
			groupedFs[z][j].Add(&groupedFs[z][j], &scaledEvaluation)
		}
	}
	g_x := make([]fr.Element, common.POLY_DEGREE)
	for index, f := range groupedFs {
		// If there is no polynomial for this evaluation point, we skip it.
		if len(f) == 0 {
			continue
		}
		quotient := ipaConf.PrecomputedWeights.DivideOnDomain(uint8(index), f)
		for j := 0; j < common.POLY_DEGREE; j++ {
			g_x[j].Add(&g_x[j], &quotient[j])
		}
	}

	D := ipaConf.Commit(g_x)

	transcript.AppendPoint(&D, "D")
	t := transcript.ChallengeScalar("t")

	// Calculate the denominator inverses only for referenced evaluation points.
	den_inv := make([]fr.Element, 0, common.POLY_DEGREE)
	for z, f := range groupedFs {
		if len(f) == 0 {
			continue
		}
		var z = domainToFr(uint8(z))
		var den fr.Element
		den.Sub(&t, &z)
		den_inv = append(den_inv, den)
	}
	den_inv = fr.BatchInvert(den_inv)

	// Compute h(X) = g_1(X)
	h_x := make([]fr.Element, common.POLY_DEGREE)
	denInvIdx := 0
	for _, f := range groupedFs {
		if len(f) == 0 {
			continue
		}
		for k := 0; k < common.POLY_DEGREE; k++ {
			var tmp fr.Element
			tmp.Mul(&f[k], &den_inv[denInvIdx])
			h_x[k].Add(&h_x[k], &tmp)
		}
		denInvIdx++
	}

	h_minus_g := make([]fr.Element, common.POLY_DEGREE)
	for i := 0; i < common.POLY_DEGREE; i++ {
		h_minus_g[i].Sub(&h_x[i], &g_x[i])
	}

	E := ipaConf.Commit(h_x)
	transcript.AppendPoint(&E, "E")

	var E_minus_D banderwagon.Element

	E_minus_D.Sub(&E, &D)

	ipa_proof := ipa.CreateIPAProof(transcript, ipaConf, E_minus_D, h_minus_g, t)

	return &MultiProof{
		IPA: ipa_proof,
		D:   D,
	}
}

func CheckMultiProof(transcript *common.Transcript, ipaConf *ipa.IPAConfig, proof *MultiProof, Cs []*banderwagon.Element, ys []*fr.Element, zs []uint8) bool {
	transcript.DomainSep("multiproof")

	if len(Cs) != len(ys) {
		panic(fmt.Sprintf("number of commitments = %d, while number of output points = %d", len(Cs), len(ys)))
	}
	if len(Cs) != len(zs) {
		panic(fmt.Sprintf("number of commitments = %d, while number of input points = %d", len(Cs), len(zs)))
	}

	num_queries := len(Cs)
	if num_queries == 0 {
		// XXX does this need to be a panic?
		// XXX: this comment is also in CreateMultiProof
		panic("cannot create a multiproof with no data")
	}

	for i := 0; i < num_queries; i++ {
		transcript.AppendPoint(Cs[i], "C")
		var z = domainToFr(zs[i])
		transcript.AppendScalar(&z, "z")
		transcript.AppendScalar(ys[i], "y")
	}
	r := transcript.ChallengeScalar("r")
	powers_of_r := common.PowersOf(r, num_queries)

	transcript.AppendPoint(&proof.D, "D")
	t := transcript.ChallengeScalar("t")

	// Compute the polynomials in lagrange form grouped by evaluation point, and
	// the needed helper scalars.
	groupedEvals := make([]fr.Element, common.POLY_DEGREE)
	for i := 0; i < num_queries; i++ {
		z := zs[i]

		// r * y_i
		r := powers_of_r[i]
		var scaledEvaluation fr.Element
		scaledEvaluation.Mul(&r, ys[i])
		groupedEvals[z].Add(&groupedEvals[z], &scaledEvaluation)
	}

	// Compute helper_scalar_den. This is 1 / t - z_i
	helper_scalar_den := make([]fr.Element, common.POLY_DEGREE)
	for i := 0; i < common.POLY_DEGREE; i++ {
		// (t - z_i)
		var z = domainToFr(uint8(i))
		helper_scalar_den[i].Sub(&t, &z)
	}
	helper_scalar_den = fr.BatchInvert(helper_scalar_den)

	// Compute g_2(t) = SUM (y_i * r^i) / (t - z_i) = SUM (y_i * r) * helper_scalars_den
	g_2_t := fr.Zero()
	for i := 0; i < common.POLY_DEGREE; i++ {
		if groupedEvals[i].IsZero() {
			continue
		}
		var tmp fr.Element
		tmp.Mul(&groupedEvals[i], &helper_scalar_den[i])
		g_2_t.Add(&g_2_t, &tmp)
	}

	// Compute E = SUM C_i * (r^i / t - z_i) = SUM C_i * msm_scalars
	msm_scalars := make([]fr.Element, len(Cs))
	Csnp := make([]banderwagon.Element, len(Cs))
	for i := 0; i < len(Cs); i++ {
		Csnp[i] = *Cs[i]

		msm_scalars[i].Mul(&powers_of_r[i], &helper_scalar_den[zs[i]])
	}
	E := ipa.MultiScalar(Csnp, msm_scalars)
	transcript.AppendPoint(&E, "E")

	var E_minus_D banderwagon.Element
	E_minus_D.Sub(&E, &proof.D)

	return ipa.CheckIPAProof(transcript, ipaConf, E_minus_D, proof.IPA, t, g_2_t)
}

func domainToFr(in uint8) fr.Element {
	var x fr.Element
	x.SetUint64(uint64(in))
	return x
}

func (mp *MultiProof) Write(w io.Writer) {
	binary.Write(w, binary.BigEndian, mp.D.Bytes())
	mp.IPA.Write(w)
}

func (mp *MultiProof) Read(r io.Reader) {
	D := common.ReadPoint(r)
	mp.D = *D
	mp.IPA.Read(r)
}
func (mp MultiProof) Equal(other MultiProof) bool {
	if !mp.IPA.Equal(other.IPA) {
		return false
	}
	return mp.D.Equal(&other.D)
}
