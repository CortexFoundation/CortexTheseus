package bandersnatch

import (
	"math/big"

	"github.com/crate-crypto/go-ipa/bandersnatch/fp"
)

// CurveParams curve parameters: ax^2 + y^2 = 1 + d*x^2*y^2
type CurveParams struct {
	A, D     fp.Element // in Montgomery form
	Cofactor fp.Element // not in Montgomery form
	Order    big.Int
	Base     PointAffine
}

var edwards CurveParams

// GetEdwardsCurve returns the twisted Edwards curve on BLS12-381's Fr
func GetEdwardsCurve() CurveParams {

	// copy to keep Order private
	var res CurveParams

	res.A.Set(&edwards.A)
	res.D.Set(&edwards.D)
	res.Cofactor.Set(&edwards.Cofactor)
	res.Order.Set(&edwards.Order)
	res.Base.Set(&edwards.Base)

	return res
}

func init() {
	// A = -5
	// bandersnatch d = 45022363124591815672509500913686876175488063829319466900776701791074614335719
	// Co-factor = 4
	// Order = 13108968793781547619861935127046491459309155893440570251786403306729687672801
	// BASE_X = 29627151942733444043031429156003786749302466371339015363120350521834195802525
	// BASE_Y = 27488387519748396681411951718153463804682561779047093991696427532072116857978
	edwards.A.SetUint64(5).Neg(&edwards.A)

	edwards.D.SetString("45022363124591815672509500913686876175488063829319466900776701791074614335719")
	edwards.Cofactor.SetUint64(4).FromMont()
	edwards.Order.SetString("13108968793781547619861935127046491459309155893440570251786403306729687672801", 10)

	edwards.Base.X.SetString("18886178867200960497001835917649091219057080094937609519140440539760939937304")
	edwards.Base.Y.SetString("19188667384257783945677642223292697773471335439753913231509108946878080696678")
}

func mulByA(x *fp.Element) {
	x.Neg(x)
	fp.MulBy5(x)
}
