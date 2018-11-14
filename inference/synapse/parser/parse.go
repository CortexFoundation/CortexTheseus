package parser

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

type Block struct {
	blkName string
	params  map[string]string
}
type Layer struct {
	layerType LayerCode
	strValue  map[UnitCode]string
	intValue  map[UnitCode]int
	outputW   int
	outputH   int
	outputC   int
}

type Parser struct {
	cfgLines []string
	size     int64
	blocks   []*Block
	layers   []*Layer
}

func (b *Block) OutputBlock() {
	fmt.Printf("Block Name : %s\n", b.blkName)
	for k, v := range b.params {
		fmt.Printf("%s : %s\n", k, v)
	}
}
func (p *Parser) ImportFile(cfgName string, modelName string) error {
	cfgF, err := os.OpenFile(cfgName, os.O_RDWR, 0666)
	if err != nil {
		return err
	}
	defer cfgF.Close()

	_, err = cfgF.Stat()
	if err != nil {
		return err
	}

	buf := bufio.NewReader(cfgF)
	p.cfgLines = make([]string, 0)
	for {
		line, err := buf.ReadString('\n')
		// line = strings.TrimSpace(line)
		if err != nil {
			if err == io.EOF {
				break
			} else {
				return err
			}
		}
		line = line[:len(line)-1]
		if line != "" {
			p.cfgLines = append(p.cfgLines, line)
			// fmt.Println(line)
		}
	}
	modelF, err := os.OpenFile(modelName, os.O_RDWR, 0666)
	if err != nil {
		return err
	}
	defer modelF.Close()

	fi, err := modelF.Stat()
	if err != nil {
		return err
	}
	p.size = fi.Size()
	return nil
}

//GenBlock load each block
func (p *Parser) GenBlock() (err error) {
	p.blocks = make([]*Block, 0)
	var b *Block = nil
	for i, str := range p.cfgLines {
		if len(str) > 2 && str[0] == '[' && str[len(str)-1] == ']' {
			b = &Block{blkName: str[1 : len(str)-1], params: make(map[string]string)}
			p.blocks = append(p.blocks, b)
		} else {
			split_equ := strings.Split(str, "=")
			if len(split_equ) != 2 {
				return errors.New(fmt.Sprintf("Missing eq op line:%d, info: %s", i, str))
			}
			_, ok := b.params[split_equ[0]]
			if ok {
				return errors.New(fmt.Sprintf("Surplus params:%d, info: %s", i, str))
			}
			b.params[split_equ[0]] = split_equ[1]
		}
	}
	return nil
}

//checkMustOnlyUnit check block name, 1st block must be "net", last block must be connected, each type of block must have reference unit
func (p *Parser) checkMustOnlyUnit() (err error) {
	p.layers = make([]*Layer, 0)
	if len(p.blocks) <= 1 {
		return errors.New("block Number is too small")
	}
	if p.blocks[0].blkName != "net" {
		return errors.New("first block must be \"net\"")
	}
	if p.blocks[len(p.blocks)-1].blkName != "connected" {
		return errors.New("last block must be \"connected\"")
	}
	for i, b := range p.blocks {
		if i != 0 && b.blkName == "net" {
			return errors.New("net must appear once and only once at the begining")
		}
		lc, ok := blkNameToCode[b.blkName]
		if !ok {
			return errors.New(fmt.Sprintf("cannot find type of block %s", b.blkName))
		}
		unitCodeList := mustOnlyUnit[lc]
		var tmpUnitList []UnitCode
		tmpUnitList = make([]UnitCode, 0)
		var l Layer
		l.layerType = lc
		l.strValue = make(map[UnitCode]string)
		l.intValue = make(map[UnitCode]int)
		for unit, value := range p.blocks[i].params {
			uc, ok := unitNameToUnitCode[unit]
			if !ok {
				return errors.New(fmt.Sprintf("cannot find unit %s", unit))
			}
			// fmt.Println(unit, b.blkName)
			tmpUnitList = append(tmpUnitList, uc)
			t, val, err := basicCheck[uc](value)
			if err != nil {
				return err
			}
			if t == INT {
				vali, ok := val.(int)
				if !ok {
					return errors.New("error type")
				}
				l.intValue[uc] = vali
			} else if t == STR {
				vals, ok := val.(string)
				if !ok {
					return errors.New("error type")
				}
				l.strValue[uc] = vals
			} else {
				return errors.New("unknown type")
			}
		}
		// fmt.Println(unitCodeList, tmpUnitList)
		err := checkMustTypeList(unitCodeList, tmpUnitList, b.blkName)
		if err != nil {
			return err
		}
		if l.layerType == LC_NET {
			l.outputC = l.intValue[UC_CHANNELS]
			l.outputW = l.intValue[UC_WIDTH]
			l.outputH = l.intValue[UC_HEIGHT]
		}
		p.layers = append(p.layers, &l)
	}
	return nil
}

//checkMustOnlyUnit check block name, 1st block must be "net", last block must be connected, each type of block must have reference unit
func (p *Parser) checkMatrixSizeAndLayer() (size int64, err error) {
	if len(p.layers) > 50 {
		return 0, errors.New("Do not support layer number larger than 50")
	}
	size = 0
	for i := 1; i < len(p.layers); i++ {
		l := p.layers[i]
		ll := p.layers[i-1]
		if ll.outputC <= 0 || ll.outputH <= 0 || ll.outputW <= 0 {
			return 0, errors.New("layer output size is wrong")
		}
		switch l.layerType {
		case LC_CONV:
			l.outputH = (ll.outputH+2*l.intValue[UC_PAD]-l.intValue[UC_SIZE])/l.intValue[UC_STRIDE] + 1
			l.outputW = (ll.outputW+2*l.intValue[UC_PAD]-l.intValue[UC_SIZE])/l.intValue[UC_STRIDE] + 1
			l.outputC = l.intValue[UC_FILTERS]
			if l.outputH <= 0 || l.outputW <= 0 || (ll.outputH+2*l.intValue[UC_PAD]-l.intValue[UC_SIZE]) <= 0 || (ll.outputW+2*l.intValue[UC_PAD]-l.intValue[UC_SIZE]) <= 0 {
				return 0, errors.New(fmt.Sprintf("Conv error, size may small than 1, layer:%d", i))
			}
			if l.outputC*l.intValue[UC_SIZE]*l.intValue[UC_SIZE] > (1 << 14) {
				return 0, errors.New("conv int32 overflow")
			}
			size += (int64)(ll.outputC)*(int64)(l.outputC)*(int64)(l.intValue[UC_SIZE])*(int64)(l.intValue[UC_SIZE]) + (int64)(l.outputC)
		case LC_ACT:
			l.outputC = ll.outputC
			l.outputW = ll.outputW
			l.outputH = ll.outputH
		case LC_MAXPOOL:
			l.outputW = (ll.outputH+0-l.intValue[UC_SIZE])/l.intValue[UC_STRIDE] + 1
			l.outputH = (ll.outputW+0-l.intValue[UC_SIZE])/l.intValue[UC_STRIDE] + 1
			l.outputC = ll.outputC
			if l.outputH <= 0 || l.outputW <= 0 || (ll.outputH-l.intValue[UC_SIZE]) <= 0 || (ll.outputW-l.intValue[UC_SIZE]) <= 0 {
				return 0, errors.New(fmt.Sprintf("maxpool error, size may small than 1, layer:%d", i))
			}
		case LC_RES:
			l.outputW = ll.outputW
			l.outputH = ll.outputH
			l.outputC = ll.outputC
			from := l.intValue[UC_FROM] + i
			if from < 0 || from >= i {
				return 0, errors.New(fmt.Sprintf("res error, from is %d, now is %d", from, i))
			}
			lll := p.layers[from]
			if lll.outputC != l.outputC || lll.outputW != l.outputW || lll.outputH != l.outputH {
				return 0, errors.New(fmt.Sprintf("res error, from layer %d, but size is not same [%d %d %d], layer %d want [%d %d %d]", from, lll.outputC, lll.outputW, lll.outputH, i, l.outputC, l.outputW, l.outputH))
			}
		case LC_FC:
			l.outputH = 1
			l.outputW = 1
			l.outputC = l.intValue[UC_OUTPUT]
			// fmt.Println(ll.outputC, ll.outputH, ll.outputW, l.outputC)
			if ll.outputC*ll.outputW*ll.outputH > (1 << 16) {
				return 0, errors.New("fc int32 overflow")
			}
			size += (int64)(ll.outputC)*(int64)(l.outputC)*(int64)(ll.outputH)*(int64)(ll.outputW) + (int64)(l.outputC)
		default:
			return 0, errors.New("error layer type")
		}
	}
	return size, nil
}

func (p *Parser) OutputBlocks() {
	for _, b := range p.blocks {
		b.OutputBlock()
		fmt.Println()
	}
}

func (p *Parser) ParseCFG() (size int64, err error) {
	err = p.GenBlock()
	if err != nil {
		return 0, err
	}
	// p.OutputBlocks()

	err = p.checkMustOnlyUnit()
	if err != nil {
		return 0, err
	}
	size, err = p.checkMatrixSizeAndLayer()
	if err != nil {
		return 0, err
	}
	// fmt.Println(size)
	return size, nil
}

func CheckModel(cfgName string, paramsName string) error {
	var p Parser
	err := p.ImportFile(cfgName, paramsName)
	if err != nil {
		return err
	}
	size, err := p.ParseCFG()
	if err != nil {
		return err
	}
	if size != p.size {
		return errors.New("model file size not match cfg")
	}
	return nil
}

func main() {
	r := CheckModel("../cfg/mnist_res_v1.cfg", "../mnist_res_v1.bin")
	if r == nil {
		fmt.Println("ok")
	} else {
		fmt.Println(r)
	}
	r = CheckModel("../cfg/dog_cat_compress_v1.cfg", "../dog_cat_compress_v1.bin")
	if r == nil {
		fmt.Println("ok")
	} else {
		fmt.Println(r)
	}
}
