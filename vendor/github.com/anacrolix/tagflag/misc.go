package tagflag

import (
	"reflect"
	"strconv"
	"unicode"

	"github.com/bradfitz/iter"
)

const flagPrefix = "-"

// Walks the fields of the given struct, calling the function with the value
// and StructField for each field. Returning true from the function will halt
// traversal.
func foreachStructField(_struct reflect.Value, f func(fv reflect.Value, sf reflect.StructField) (stop bool)) {
	t := _struct.Type()
	for i := range iter.N(t.NumField()) {
		sf := t.Field(i)
		fv := _struct.Field(i)
		if f(fv, sf) {
			break
		}
	}
}

func canMarshal(f reflect.Value) bool {
	return valueMarshaler(f.Type()) != nil
}

// Returns a marshaler for the given value, or nil if there isn't one.
func valueMarshaler(t reflect.Type) marshaler {
	if zm, ok := reflect.Zero(reflect.PtrTo(t)).Interface().(Marshaler); ok {
		return dynamicMarshaler{
			marshal: func(v reflect.Value, s string) error {
				return v.Addr().Interface().(Marshaler).Marshal(s)
			},
			explicitValueRequired: zm.RequiresExplicitValue(),
		}
	}
	if bm, ok := builtinMarshalers[t]; ok {
		return bm
	}
	switch t.Kind() {
	case reflect.Ptr:
		m := valueMarshaler(t.Elem())
		if m == nil {
			return nil
		}
		return ptrMarshaler{m}
	case reflect.Struct:
		return nil
	case reflect.Bool:
		return dynamicMarshaler{
			marshal: func(v reflect.Value, s string) error {
				if s == "" {
					v.SetBool(true)
					return nil
				}
				b, err := strconv.ParseBool(s)
				v.SetBool(b)
				return err
			},
			explicitValueRequired: false,
		}
	}
	return defaultMarshaler{}
}

// Turn a struct field name into a flag name. In particular this lower cases
// leading acronyms, and the first capital letter.
func fieldFlagName(fieldName string) flagNameComponent {
	return flagNameComponent(func() (ret []rune) {
		fieldNameRunes := []rune(fieldName)
		for i, r := range fieldNameRunes {
			prevUpper := func() bool { return unicode.IsUpper(fieldNameRunes[i-1]) }
			nextUpper := func() bool { return unicode.IsUpper(fieldNameRunes[i+1]) }
			if i == 0 || (prevUpper() && (i == len(fieldNameRunes)-1 || nextUpper())) {
				r = unicode.ToLower(r)
			}
			ret = append(ret, r)
		}
		return
	}())
}
