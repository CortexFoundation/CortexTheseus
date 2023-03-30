package banderwagon

import (
	"github.com/crate-crypto/go-ipa/bandersnatch"
	"github.com/crate-crypto/go-ipa/bandersnatch/fr"
)

// MultiExpConfig enables to set optional configuration attribute to a call to MultiExp
type MultiExpConfig struct {
	NbTasks     int  // go routines to be used in the multiexp. can be larger than num cpus.
	ScalarsMont bool // indicates if the scalars are in montgomery form. Default to false.
}

func (p *Element) MultiExp(points []Element, scalars []fr.Element, _config MultiExpConfig) (*Element, error) {
	var pointsAffs = make([]bandersnatch.PointAffine, len(points))
	for i := 0; i < len(points); i++ {
		// TODO: improve speed by using Montgomery batch normalisation algorithm
		var AffRepr bandersnatch.PointAffine
		AffRepr.FromProj(&points[i].inner)
		pointsAffs[i] = AffRepr
	}

	config := bandersnatch.MultiExpConfig{
		NbTasks:     _config.NbTasks,
		ScalarsMont: _config.ScalarsMont,
	}
	// NOTE: This is fine as long MultiExp does not use Equal functionality
	_, err := p.inner.MultiExp(pointsAffs, scalars, config)

	return p, err
}
