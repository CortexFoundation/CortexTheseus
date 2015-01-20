package qt

import (
	"github.com/ethereum/go-ethereum/core"
	"github.com/ethereum/go-ethereum/ui"
	"gopkg.in/qml.v1"
)

func NewFilterFromMap(object map[string]interface{}, eth core.EthManager) *core.Filter {
	filter := ui.NewFilterFromMap(object, eth)

	if object["altered"] != nil {
		filter.Altered = makeAltered(object["altered"])
	}

	return filter
}

func makeAltered(v interface{}) (d []core.AccountChange) {
	if qList, ok := v.(*qml.List); ok {
		var s []interface{}
		qList.Convert(&s)

		d = makeAltered(s)
	} else if qMap, ok := v.(*qml.Map); ok {
		var m map[string]interface{}
		qMap.Convert(&m)

		d = makeAltered(m)
	}

	return
}
