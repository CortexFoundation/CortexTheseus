package netutil

import "net"

// AddrIP gets the IP address contained in addr. It returns nil if no address is present.
func AddrIP(addr net.Addr) net.IP {
	switch a := addr.(type) {
	case *net.IPAddr:
		return a.IP
	case *net.TCPAddr:
		return a.IP
	case *net.UDPAddr:
		return a.IP
	default:
		return nil
	}
}
