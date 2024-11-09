// Copyright 2020 The go-ethereum Authors
// This file is part of The go-ethereum library.
//
// The go-ethereum library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The go-ethereum library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The go-ethereum library. If not, see <http://www.gnu.org/licenses/>.

package node

import (
	"compress/gzip"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"

	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/rs/cors"
)

// NewHTTPHandlerStack returns wrapped http-related handlers
func NewHTTPHandlerStack(srv http.Handler, cors []string, vhosts []string) http.Handler {
	// Wrap the CORS-handler within a host-handler
	handler := newCorsHandler(srv, cors)
	handler = newVHostHandler(vhosts, handler)
	return newGzipHandler(handler)
}

func newCorsHandler(srv http.Handler, allowedOrigins []string) http.Handler {
	// disable CORS support if user has not specified a custom CORS configuration
	if len(allowedOrigins) == 0 {
		return srv
	}
	c := cors.New(cors.Options{
		AllowedOrigins: allowedOrigins,
		AllowedMethods: []string{http.MethodPost, http.MethodGet},
		MaxAge:         600,
		AllowedHeaders: []string{"*"},
	})
	return c.Handler(srv)
}

// virtualHostHandler is a handler which validates the Host-header of incoming requests.
// Using virtual hosts can help prevent DNS rebinding attacks, where a 'random' domain name points to
// the service ip address (but without CORS headers). By verifying the targeted virtual host, we can
// ensure that it's a destination that the node operator has defined.
type virtualHostHandler struct {
	vhosts map[string]struct{}
	next   http.Handler
}

func newVHostHandler(vhosts []string, next http.Handler) http.Handler {
	vhostMap := make(map[string]struct{})
	for _, allowedHost := range vhosts {
		vhostMap[strings.ToLower(allowedHost)] = struct{}{}
	}
	return &virtualHostHandler{vhostMap, next}
}

// ServeHTTP serves JSON-RPC requests over HTTP, implements http.Handler
func (h *virtualHostHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// if r.Host is not set, we can continue serving since a browser would set the Host header
	if r.Host == "" {
		h.next.ServeHTTP(w, r)
		return
	}
	host, _, err := net.SplitHostPort(r.Host)
	if err != nil {
		// Either invalid (too many colons) or no port specified
		host = r.Host
	}
	if ipAddr := net.ParseIP(host); ipAddr != nil {
		// It's an IP address, we can serve that
		h.next.ServeHTTP(w, r)
		return

	}
	// Not an IP address, but a hostname. Need to validate
	if _, exist := h.vhosts["*"]; exist {
		h.next.ServeHTTP(w, r)
		return
	}
	if _, exist := h.vhosts[host]; exist {
		h.next.ServeHTTP(w, r)
		return
	}
	http.Error(w, "invalid host specified", http.StatusForbidden)
}

var gzPool = sync.Pool{
	New: func() any {
		w := gzip.NewWriter(io.Discard)
		return w
	},
}

type gzipResponseWriter struct {
	resp http.ResponseWriter

	gz            *gzip.Writer
	contentLength uint64 // total length of the uncompressed response
	written       uint64 // amount of written bytes from the uncompressed response
	hasLength     bool   // true if uncompressed response had Content-Length
	inited        bool   // true after init was called for the first time
}

// init runs just before response headers are written. Among other things, this function
// also decides whether compression will be applied at all.
func (w *gzipResponseWriter) init() {
	if w.inited {
		return
	}
	w.inited = true

	hdr := w.resp.Header()
	length := hdr.Get("content-length")
	if len(length) > 0 {
		if n, err := strconv.ParseUint(length, 10, 64); err != nil {
			w.hasLength = true
			w.contentLength = n
		}
	}

	// Setting Transfer-Encoding to "identity" explicitly disables compression. net/http
	// also recognizes this header value and uses it to disable "chunked" transfer
	// encoding, trimming the header from the response. This means downstream handlers can
	// set this without harm, even if they aren't wrapped by newGzipHandler.
	//
	// In CortexTheseus, we use this signal to disable compression for certain error
	// responses which are flushed out close to the write deadline of the response. For
	// these cases, we want to avoid chunked transfer encoding and compression because
	// they require additional output that may not get written in time.
	passthrough := hdr.Get("transfer-encoding") == "identity"
	if !passthrough {
		w.gz = gzPool.Get().(*gzip.Writer)
		w.gz.Reset(w.resp)
		hdr.Del("content-length")
		hdr.Set("content-encoding", "gzip")
	}
}

func (w *gzipResponseWriter) Header() http.Header {
	return w.resp.Header()
}

func (w *gzipResponseWriter) WriteHeader(status int) {
	w.init()
	w.resp.WriteHeader(status)
}

func (w *gzipResponseWriter) Write(b []byte) (int, error) {
	w.init()

	if w.gz == nil {
		// Compression is disabled.
		return w.resp.Write(b)
	}

	n, err := w.gz.Write(b)
	w.written += uint64(n)
	if w.hasLength && w.written >= w.contentLength {
		// The HTTP handler has finished writing the entire uncompressed response. Close
		// the gzip stream to ensure the footer will be seen by the client in case the
		// response is flushed after this call to write.
		err = w.gz.Close()
	}
	return n, err
}

func (w *gzipResponseWriter) Flush() {
	if w.gz != nil {
		w.gz.Flush()
	}
	if f, ok := w.resp.(http.Flusher); ok {
		f.Flush()
	}
}

func (w *gzipResponseWriter) close() {
	if w.gz == nil {
		return
	}
	w.gz.Close()
	gzPool.Put(w.gz)
	w.gz = nil
}

func newGzipHandler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			next.ServeHTTP(w, r)
			return
		}

		wrapper := &gzipResponseWriter{resp: w}
		defer wrapper.close()

		next.ServeHTTP(wrapper, r)
	})
}

// NewWebsocketUpgradeHandler returns a websocket handler that serves an incoming request only if it contains an upgrade
// request to the websocket protocol. If not, serves the request with the http handler.
func NewWebsocketUpgradeHandler(h http.Handler, ws http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isWebsocket(r) {
			ws.ServeHTTP(w, r)
			log.Debug("serving websocket request")
			return
		}

		h.ServeHTTP(w, r)
	})
}

// isWebsocket checks the header of an http request for a websocket upgrade request.
func isWebsocket(r *http.Request) bool {
	return strings.ToLower(r.Header.Get("Upgrade")) == "websocket" &&
		strings.Contains(strings.ToLower(r.Header.Get("Connection")), "upgrade")
}
