package tagflag

import (
	"fmt"
	"reflect"
	"strconv"

	"github.com/pkg/errors"
)

type Marshaler interface {
	Marshal(in string) error
	// Must have and ignore a pointer receiver.
	RequiresExplicitValue() bool
}

type marshaler interface {
	Marshal(reflect.Value, string) error
	RequiresExplicitValue() bool
}

type dynamicMarshaler struct {
	explicitValueRequired bool
	marshal               func(reflect.Value, string) error
}

func (me dynamicMarshaler) Marshal(v reflect.Value, s string) error {
	return me.marshal(v, s)
}

func (me dynamicMarshaler) RequiresExplicitValue() bool {
	return me.explicitValueRequired
}

// The fallback marshaler, that attempts to use fmt.Sscan, and recursion to
// sort marshal types.
type defaultMarshaler struct{}

func (defaultMarshaler) Marshal(v reflect.Value, s string) error {
	switch v.Kind() {
	case reflect.Slice:
		n := reflect.New(v.Type().Elem())
		m := valueMarshaler(n.Elem().Type())
		if m == nil {
			return fmt.Errorf("can't marshal type %s", n.Elem().Type())
		}
		err := m.Marshal(n.Elem(), s)
		if err != nil {
			return err
		}
		v.Set(reflect.Append(v, n.Elem()))
		return nil
	case reflect.Int:
		x, err := strconv.ParseInt(s, 0, 0)
		v.SetInt(x)
		return err
	case reflect.Uint:
		x, err := strconv.ParseUint(s, 0, 0)
		v.SetUint(x)
		return err
	case reflect.Int64:
		x, err := strconv.ParseInt(s, 0, 64)
		v.SetInt(x)
		return err
	case reflect.String:
		v.SetString(s)
		return nil
	case reflect.Array:
		if v.Type().Elem().Kind() == reflect.Uint8 {
			if len(s) == 2*v.Len() {
				var sl []byte = v.Slice(0, v.Len()).Interface().([]byte)
				// log.Println(cap(sl), len(sl))
				// hex.DecodeString(s)
				_, err := fmt.Sscanf(s, "%x", &sl)
				// log.Println(cap(sl), sl, err)
				if err != nil {
					return errors.Wrapf(err, "scanning hex")
				}
				// Seems the slice moves as part of the decoding :|
				reflect.Copy(v, reflect.ValueOf(sl))
			} else {
				return errors.Errorf("argument has unhandled length %d", len(s))
			}
			return nil
		} else {
			return errors.Errorf("unhandled array elem type: %s", v.Type().String())
		}
	default:
		return fmt.Errorf("unhandled builtin type: %s", v.Type().String())
	}
}

func (defaultMarshaler) RequiresExplicitValue() bool {
	return true
}

type ptrMarshaler struct {
	inner marshaler
}

func (me ptrMarshaler) Marshal(v reflect.Value, s string) error {
	elemValue := reflect.New(v.Type().Elem())
	v.Set(elemValue)
	return me.inner.Marshal(elemValue.Elem(), s)
}

func (me ptrMarshaler) RequiresExplicitValue() bool {
	return false
}
