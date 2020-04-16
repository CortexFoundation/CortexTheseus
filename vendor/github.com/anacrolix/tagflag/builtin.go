package tagflag

import (
	"errors"
	"net"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"

	"golang.org/x/xerrors"
)

var builtinMarshalers = map[reflect.Type]marshaler{}

// Convenience function to allow adding  marshalers using typed functions.
// marshalFunc is of type func(arg string) T or func(arg string) (T, error),
// where T is the type the function can marshal.
func addBuiltinDynamicMarshaler(marshalFunc interface{}, explicitValueRequired bool) {
	marshalFuncValue := reflect.ValueOf(marshalFunc)
	marshalType := marshalFuncValue.Type().Out(0)
	builtinMarshalers[marshalType] = dynamicMarshaler{
		marshal: func(marshalValue reflect.Value, arg string) error {
			out := marshalFuncValue.Call([]reflect.Value{reflect.ValueOf(arg)})
			marshalValue.Set(out[0])
			if len(out) > 1 {
				i := out[1].Interface()
				if i != nil {
					return i.(error)
				}
			}
			return nil
		},
		explicitValueRequired: explicitValueRequired,
	}
}

func init() {
	// These are some simple builtin types that are nice to be handled without
	// wrappers that implement Marshaler. Note that if they return pointer
	// types, those must be used in the flag struct, because there's no way to
	// know that nothing depends on the address returned.
	addBuiltinDynamicMarshaler(func(urlStr string) (*url.URL, error) {
		return url.Parse(urlStr)
	}, false)
	// Empty strings for this type are valid, so we enforce that the value is
	// explicit (=), so that the user knows what they're getting into.
	addBuiltinDynamicMarshaler(func(s string) (*net.TCPAddr, error) {
		if s == "" {
			return nil, nil
		}
		var ret net.TCPAddr
		retAlt, err := parseIpPortZone(s)
		if err != nil {
			return nil, err
		}
		ret = net.TCPAddr(retAlt)
		return &ret, err
	}, true)
	addBuiltinDynamicMarshaler(func(s string) (time.Duration, error) {
		return time.ParseDuration(s)
	}, false)
	addBuiltinDynamicMarshaler(func(s string) net.IP {
		return net.ParseIP(s)
	}, false)
}

func parseIpAddr(host string) (ret net.IPAddr, err error) {
	ss := strings.SplitN(host, "%", 2)
	ret.IP = net.ParseIP(ss[0])
	if ret.IP == nil && ss[0] != "" {
		err = errors.New("error parsing IP")
		return
	}
	if len(ss) >= 2 {
		ret.Zone = ss[1]
	}
	return
}

type ipPortZone struct {
	IP   net.IP
	Port int
	Zone string
}

func parseIpPortZone(hostport string) (ret ipPortZone, err error) {
	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		return
	}
	portInt64, err := strconv.ParseInt(port, 10, 0)
	if err != nil {
		return
	}
	ret.Port = int(portInt64)
	ipAddr, err := parseIpAddr(host)
	if err != nil {
		err = xerrors.Errorf("parsing host %q: %w", host, err)
		return
	}
	ret.IP = ipAddr.IP
	ret.Zone = ipAddr.Zone
	return
}
