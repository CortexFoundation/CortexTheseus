package parser

import (
	"errors"
	"fmt"
)

func unitErrGen(unitName string, mustbe string, now interface{}) error {
	switch now.(type) {
	case string:
		res, _ := now.(string)
		return errors.New(fmt.Sprintf("Unit [%s] must be [%s], now is [%s]", unitName, mustbe, res))
	case int:
		res, _ := now.(int)
		return errors.New(fmt.Sprintf("Unit [%s] must be [%s], now is [%d]", unitName, mustbe, res))

	default:
		return errors.New("error with now type")
	}
}
func checkMustTypeList(listOrigin []UnitCode, listCfg []UnitCode, blkName string) error {
	for _, codeO := range listOrigin {
		find := false
		for _, codeT := range listCfg {
			if codeT == codeO {
				find = true
				break
			}
		}
		if !find {
			return errors.New(fmt.Sprintf("code %s cannot find in %s", unitCodeToString[codeO], blkName))
		}
	}
	for _, codeT := range listCfg {
		find := false
		for _, codeO := range listOrigin {
			if codeT == codeO {
				find = true
				break
			}
		}
		if !find {
			return errors.New(fmt.Sprintf("code %s in block %s  do not need", unitCodeToString[codeT], blkName))
		}
	}
	return nil
}
