package parser

import (
	"strconv"
)

const (
	LC_NET LayerCode = iota
	LC_CONV
	LC_ACT
	LC_FC
	LC_RES
	LC_MAXPOOL
)
const (
	UC_BATCH UnitCode = iota
	UC_SUBDIVISIONS
	UC_HEIGHT
	UC_WIDTH
	UC_CHANNELS
	UC_FILTERS
	UC_SIZE
	UC_STRIDE
	UC_PAD
	UC_ACTIVATION
	UC_OUTPUT
	UC_FROM
)

var layerCodeToString = map[LayerCode]string{
	LC_NET:     "net",
	LC_CONV:    "convolutional",
	LC_ACT:     "activation",
	LC_FC:      "connected",
	LC_RES:     "shortcut",
	LC_MAXPOOL: "maxpool",
}
var unitCodeToString = map[UnitCode]string{
	UC_BATCH:        "batch",
	UC_SUBDIVISIONS: "subdivisions",
	UC_HEIGHT:       "height",
	UC_WIDTH:        "width",
	UC_CHANNELS:     "channels",
	UC_FILTERS:      "filters",
	UC_SIZE:         "size",
	UC_STRIDE:       "stride",
	UC_PAD:          "pad",
	UC_ACTIVATION:   "activation",
	UC_OUTPUT:       "output",
	UC_FROM:         "from",
}

var unitNameToUnitCode = map[string]UnitCode{
	"batch":        UC_BATCH,
	"subdivisions": UC_SUBDIVISIONS,
	"height":       UC_HEIGHT,
	"width":        UC_WIDTH,
	"channels":     UC_CHANNELS,
	"filters":      UC_FILTERS,
	"size":         UC_SIZE,
	"stride":       UC_STRIDE,
	"pad":          UC_PAD,
	"activation":   UC_ACTIVATION,
	"output":       UC_OUTPUT,
	"from":         UC_FROM,
}

const (
	STR ValueType = iota
	INT
	ERR
)

var blkNameToCode = map[string]LayerCode{
	"net":           LC_NET,
	"convolutional": LC_CONV,
	"activation":    LC_ACT,
	"connected":     LC_FC,
	"shortcut":      LC_RES,
	"maxpool":       LC_MAXPOOL,
}

type UnitCode byte
type LayerCode byte
type ValueType byte

var mustOnlyUnit = map[LayerCode][]UnitCode{
	LC_NET: {
		UC_BATCH,
		UC_SUBDIVISIONS,
		UC_HEIGHT,
		UC_WIDTH,
		UC_CHANNELS,
	},
	LC_CONV: {
		UC_FILTERS,
		UC_SIZE,
		UC_STRIDE,
		UC_PAD,
		UC_ACTIVATION,
	},
	LC_ACT: {
		UC_ACTIVATION,
	},
	LC_FC: {
		UC_OUTPUT,
		UC_ACTIVATION,
	},
	LC_RES: {
		UC_FROM,
	},
	LC_MAXPOOL: {
		UC_SIZE,
		UC_STRIDE,
	},
}

type (
	basicCheckFunc func(string) (ValueType, interface{}, error)
)

func basicActivationCheck(value string) (ValueType, interface{}, error) {
	if (value != "relu") && (value != "linear") {
		return ERR, nil, unitErrGen("activation", "linear | relu", value)
	}
	return STR, value, nil
}
func basicBatchCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res != 1 {
		return ERR, nil, unitErrGen("batch", "1", res)
	}
	return INT, res, nil
}
func basicSubdivisionsCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res != 1 {
		return ERR, nil, unitErrGen("subdivisions", "1", res)
	}
	return INT, res, nil
}
func basicHeightCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res < 1 || res > 256 {
		return ERR, nil, unitErrGen("height", "1~256", res)
	}
	return INT, res, nil
}
func basicWidthCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res < 1 || res > 256 {
		return ERR, nil, unitErrGen("height", "1~256", res)
	}
	return INT, res, nil
}
func basicChannelCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res != 1 && res != 3 {
		return ERR, nil, unitErrGen("channel", "1|3", res)
	}
	return INT, res, nil
}
func basicFiltersCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res < 1 || res > 512 {
		return ERR, nil, unitErrGen("filters", "1~512", res)
	}
	return INT, res, nil
}
func basicSizeCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res < 1 || res > 256 {
		return ERR, nil, unitErrGen("size", "1~256", res)
	}
	return INT, res, nil
}
func basicStrideCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res < 1 || res > 256 {
		return ERR, nil, unitErrGen("stride", "1~256", res)
	}
	return INT, res, nil
}
func basicPadCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res < 0 || res > 256 {
		return ERR, nil, unitErrGen("pad", "0~256", res)
	}
	return INT, res, nil
}
func basicOutputCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res < 1 {
		return ERR, nil, unitErrGen("output", "larger than 1", res)
	}
	return INT, res, nil
}
func basicFromCheck(value string) (ValueType, interface{}, error) {
	res, err := strconv.Atoi(value)
	if err != nil {
		return ERR, nil, err
	}
	if res > 0 {
		return ERR, nil, unitErrGen("from", "negative", res)
	}
	return INT, res, nil
}

var basicCheck = map[UnitCode]basicCheckFunc{
	UC_BATCH:        basicBatchCheck,
	UC_SUBDIVISIONS: basicSubdivisionsCheck,
	UC_HEIGHT:       basicHeightCheck,
	UC_WIDTH:        basicWidthCheck,
	UC_CHANNELS:     basicChannelCheck,
	UC_FILTERS:      basicFiltersCheck,
	UC_SIZE:         basicSizeCheck,
	UC_STRIDE:       basicStrideCheck,
	UC_PAD:          basicPadCheck,
	UC_OUTPUT:       basicOutputCheck,
	UC_FROM:         basicFromCheck,
	UC_ACTIVATION:   basicActivationCheck,
}
