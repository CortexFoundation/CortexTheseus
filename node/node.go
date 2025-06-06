// Copyright 2015 The go-ethereum Authors
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
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"

	"github.com/CortexFoundation/CortexTheseus/accounts"
	"github.com/CortexFoundation/CortexTheseus/core/rawdb"
	"github.com/CortexFoundation/CortexTheseus/ctxcdb"
	"github.com/CortexFoundation/CortexTheseus/ctxcdb/memorydb"
	"github.com/CortexFoundation/CortexTheseus/event"
	"github.com/CortexFoundation/CortexTheseus/internal/debug"
	"github.com/CortexFoundation/CortexTheseus/log"
	"github.com/CortexFoundation/CortexTheseus/p2p"
	"github.com/CortexFoundation/CortexTheseus/rpc"
	"github.com/gofrs/flock"
)

// Node is a container on which services can be registered.
type Node struct {
	eventmux *event.TypeMux // Event multiplexer used between the services of a stack
	config   *Config
	accman   *accounts.Manager

	ephemeralKeystore string       // if non-empty, the key directory that will be removed by Stop
	dirLock           *flock.Flock // prevents concurrent use of instance directory

	serverConfig p2p.Config
	server       *p2p.Server // Currently running P2P networking layer

	startStopLock sync.Mutex // Start/Stop are protected by an additional lock
	state         int        // Tracks state of node lifecycle

	serviceFuncs []ServiceConstructor     // Service constructors (in dependency order)
	services     map[reflect.Type]Service // Currently running services

	rpcAPIs       []rpc.API   // List of APIs currently provided by the node
	inprocHandler *rpc.Server // In-process RPC request handler to process the API requests

	ipcEndpoint string       // IPC endpoint to listen at (empty = IPC disabled)
	ipcListener net.Listener // IPC RPC listener socket to serve API requests
	ipcHandler  *rpc.Server  // IPC RPC request handler to process the API requests

	httpEndpoint     string       // HTTP endpoint (interface + port) to listen at (empty = HTTP disabled)
	httpWhitelist    []string     // HTTP RPC modules to allow through this endpoint
	httpListenerAddr net.Addr     // Address of HTTP RPC listener socket serving API requests
	httpServer       *http.Server // HTTP RPC HTTP server
	httpHandler      *rpc.Server  // HTTP RPC request handler to process the API requests

	wsEndpoint     string       // WebSocket endpoint (interface + port) to listen at (empty = WebSocket disabled)
	wsListenerAddr net.Addr     // Address of WebSocket RPC listener socket serving API requests
	wsHTTPServer   *http.Server // WebSocket RPC HTTP server
	wsHandler      *rpc.Server  // WebSocket RPC request handler to process the API requests

	stop chan struct{} // Channel to wait for termination notifications
	lock sync.RWMutex

	log       log.Logger
	databases map[*closeTrackingDB]struct{} // All open databases
}

const (
	initializingState = iota
	runningState
	closedState
)

// New creates a new P2P node, ready for protocol registration.
func New(conf *Config) (*Node, error) {
	// Copy config and resolve the datadir so future changes to the current
	// working directory don't affect the node.
	confCopy := *conf
	conf = &confCopy
	if conf.DataDir != "" {
		absdatadir, err := filepath.Abs(conf.DataDir)
		if err != nil {
			return nil, err
		}
		conf.DataDir = absdatadir
	}
	// Ensure that the instance name doesn't cause weird conflicts with
	// other files in the data directory.
	if strings.ContainsAny(conf.Name, `/\`) {
		return nil, errors.New(`Config.Name must not contain '/' or '\'`)
	}
	if conf.Name == datadirDefaultKeyStore {
		return nil, errors.New(`Config.Name cannot be "` + datadirDefaultKeyStore + `"`)
	}
	if strings.HasSuffix(conf.Name, ".ipc") {
		return nil, errors.New(`Config.Name cannot end in ".ipc"`)
	}
	// Ensure that the AccountManager method works before the node has started.
	// We rely on this in cmd/cortex.
	am, ephemeralKeystore, err := makeAccountManager(conf)
	if err != nil {
		return nil, err
	}
	if conf.Logger == nil {
		conf.Logger = log.New()
	}
	// Note: any interaction with Config that would create/touch files
	// in the data directory or instance directory is delayed until Start.
	node := &Node{
		accman:            am,
		ephemeralKeystore: ephemeralKeystore,
		config:            conf,
		serviceFuncs:      []ServiceConstructor{},
		ipcEndpoint:       conf.IPCEndpoint(),
		httpEndpoint:      conf.HTTPEndpoint(),
		wsEndpoint:        conf.WSEndpoint(),
		eventmux:          new(event.TypeMux),
		log:               conf.Logger,
		stop:              make(chan struct{}),
		server:            &p2p.Server{Config: conf.P2P},
		databases:         make(map[*closeTrackingDB]struct{}),
	}

	// Initialize the p2p server. This creates the node key and discovery databases.
	node.server.Config.PrivateKey = node.config.NodeKey()
	node.server.Config.Name = node.config.NodeName()
	node.server.Config.Logger = node.log
	if node.server.Config.NodeDatabase == "" {
		node.server.Config.NodeDatabase = node.config.NodeDB()
	}
	if node.server.Config.StaticNodes == nil {
		node.server.Config.StaticNodes = node.config.StaticNodes()
	}
	if node.serverConfig.TrustedNodes == nil {
		node.server.Config.TrustedNodes = node.config.TrustedNodes()
	}

	return node, nil
}

// Close stops the Node and releases resources acquired in
// Node constructor New.
func (n *Node) Close() error {
	var errs []error

	if n.state == closedState {
		return ErrNodeStopped
	}

	// Terminate all subsystems and collect any errors
	if err := n.Stop(); err != nil && err != ErrNodeStopped {
		errs = append(errs, err)
	}

	if err := n.accman.Close(); err != nil {
		errs = append(errs, err)
	}

	close(n.stop)

	// Report any errors that might have occurred
	switch len(errs) {
	case 0:
		return nil
	case 1:
		return errs[0]
	default:
		return fmt.Errorf("%v", errs)
	}
}

// Register injects a new service into the node's stack. The service created by
// the passed constructor must be unique in its type with regard to sibling ones.
func (n *Node) Register(constructor ServiceConstructor) error {
	n.lock.Lock()
	defer n.lock.Unlock()

	//if n.server != nil {
	if n.state == runningState {
		return ErrNodeRunning
	}
	n.serviceFuncs = append(n.serviceFuncs, constructor)
	return nil
}

// Start creates a live P2P node and starts running it.
func (n *Node) Start() error {
	n.startStopLock.Lock()
	defer n.startStopLock.Unlock()

	n.lock.Lock()
	defer n.lock.Unlock()

	// Short circuit if the node's already running
	//if n.server != nil {
	if n.state == runningState {
		return ErrNodeRunning
	}
	n.state = runningState
	if err := n.openDataDir(); err != nil {
		return err
	}

	// Initialize the p2p server. This creates the node key and
	// discovery databases.
	/*n.serverConfig = n.config.P2P
	n.serverConfig.PrivateKey = n.config.NodeKey()
	n.serverConfig.Name = n.config.NodeName()
	n.serverConfig.Logger = n.log
	if n.serverConfig.StaticNodes == nil {
		n.serverConfig.StaticNodes = n.config.StaticNodes()
	}
	if n.serverConfig.TrustedNodes == nil {
		n.serverConfig.TrustedNodes = n.config.TrustedNodes()
	}
	if n.serverConfig.NodeDatabase == "" {
		n.serverConfig.NodeDatabase = n.config.NodeDB()
	}*/
	running := n.server //&p2p.Server{Config: n.serverConfig}
	n.log.Info("Starting peer-to-peer node", "instance", n.serverConfig.Name)

	// Otherwise copy and specialize the P2P configuration
	services := make(map[reflect.Type]Service)
	for _, constructor := range n.serviceFuncs {
		// Create a new context for the particular service
		ctx := &ServiceContext{
			//Config:         *n.config,
			services: make(map[reflect.Type]Service),
			//EventMux:       n.eventmux,
			//AccountManager: n.accman,
		}
		for kind, s := range services { // copy needed for threaded access
			ctx.services[kind] = s
		}
		// Construct and save the service
		service, err := constructor(ctx)
		if err != nil {
			log.Error("n.serviceFuncs", "err", err)
			return err
		}
		kind := reflect.TypeOf(service)
		if _, exists := services[kind]; exists {
			return &DuplicateServiceError{Kind: kind}
		}
		services[kind] = service
	}
	// Gather the protocols and start the freshly assembled P2P server
	for _, service := range services {
		running.Protocols = append(running.Protocols, service.Protocols()...)
	}
	if err := running.Start(); err != nil {
		return convertFileLockError(err)
	}
	// Start each of the services
	var started []reflect.Type
	for kind, service := range services {
		// Start the next service, stopping all previous upon failure
		if err := service.Start(); err != nil {
			for _, kind := range started {
				services[kind].Stop()
			}
			running.Stop()

			return err
		}
		// Mark the service started for potential cleanup
		started = append(started, kind)
	}
	// Lastly, start the configured RPC interfaces
	if err := n.startRPC(services); err != nil {
		for _, service := range services {
			service.Stop()
		}
		running.Stop()
		return err
	}
	// Finish initializing the startup
	n.services = services
	//n.server = running
	//n.stop = make(chan struct{})
	return nil
}

// RegisterAPIs registers the APIs a service provides on the node.
func (n *Node) RegisterAPIs(apis []rpc.API) {
	n.lock.Lock()
	defer n.lock.Unlock()
	log.Warn("Register api", "api", apis)

	if n.state != initializingState {
		panic("can't register APIs on running/stopped node")
	}
	n.rpcAPIs = append(n.rpcAPIs, apis...)
	log.Warn("Register api end", "api", apis, "n.rpcAPIs", n.rpcAPIs)
}

// Config returns the configuration of node.
func (n *Node) Config() *Config {
	return n.config
}

func (n *Node) openDataDir() error {
	if n.config.DataDir == "" {
		return nil // ephemeral
	}

	instdir := filepath.Join(n.config.DataDir, n.config.name())
	if err := os.MkdirAll(instdir, 0700); err != nil {
		return err
	}
	// Lock the instance directory to prevent concurrent use by another instance as well as
	// accidental use of the instance directory as a database.
	n.dirLock = flock.New(filepath.Join(instdir, "LOCK"))

	if locked, err := n.dirLock.TryLock(); err != nil {
		return err
	} else if !locked {
		return ErrDatadirUsed
	}
	return nil
}

func (n *Node) closeDataDir() {
	// Release instance directory lock.
	if n.dirLock != nil && n.dirLock.Locked() {
		n.dirLock.Unlock()
		n.dirLock = nil
	}
}

// startRPC is a helper method to start all the various RPC endpoints during node
// startup. It's not meant to be called at any time afterwards as it makes certain
// assumptions about the state of the node.
func (n *Node) startRPC(services map[reflect.Type]Service) error {
	// Gather all the possible APIs to surface
	apis := n.apis()
	for _, service := range services {
		apis = append(apis, service.APIs()...)
	}
	// Start the various API endpoints, terminating all in case of errors
	if err := n.startInProc(apis); err != nil {
		return err
	}
	if err := n.startIPC(apis); err != nil {
		n.stopInProc()
		return err
	}
	if err := n.startHTTP(n.httpEndpoint, apis, n.config.HTTPModules, n.config.HTTPCors, n.config.HTTPVirtualHosts, n.config.HTTPTimeouts, n.config.WSOrigins); err != nil {
		n.stopIPC()
		n.stopInProc()
		return err
	}
	// if endpoints are not the same, start separate servers
	if n.httpEndpoint != n.wsEndpoint {
		if err := n.startWS(n.wsEndpoint, apis, n.config.WSModules, n.config.WSOrigins, n.config.WSExposeAll); err != nil {
			n.stopHTTP()
			n.stopIPC()
			n.stopInProc()
			return err
		}
	}

	// All API endpoints started successfully
	n.rpcAPIs = apis
	return nil
}

// startInProc initializes an in-process RPC endpoint.
func (n *Node) startInProc(apis []rpc.API) error {
	// Register all the APIs exposed by the services
	handler := rpc.NewServer()
	for _, api := range apis {
		if err := handler.RegisterName(api.Namespace, api.Service); err != nil {
			return err
		}
		n.log.Debug("InProc registered", "namespace", api.Namespace)
	}
	n.inprocHandler = handler
	return nil
}

// stopInProc terminates the in-process RPC endpoint.
func (n *Node) stopInProc() {
	if n.inprocHandler != nil {
		n.inprocHandler.Stop()
		n.inprocHandler = nil
	}
}

// startIPC initializes and starts the IPC RPC endpoint.
func (n *Node) startIPC(apis []rpc.API) error {
	if n.ipcEndpoint == "" {
		return nil // IPC disabled.
	}
	listener, handler, err := rpc.StartIPCEndpoint(n.ipcEndpoint, apis)
	if err != nil {
		return err
	}
	n.ipcListener = listener
	n.ipcHandler = handler
	n.log.Info("IPC endpoint opened", "url", n.ipcEndpoint)
	return nil
}

// stopIPC terminates the IPC RPC endpoint.
func (n *Node) stopIPC() {
	if n.ipcListener != nil {
		n.ipcListener.Close()
		n.ipcListener = nil

		n.log.Info("IPC endpoint closed", "url", n.ipcEndpoint)
	}
	if n.ipcHandler != nil {
		n.ipcHandler.Stop()
		n.ipcHandler = nil
	}
}

// startHTTP initializes and starts the HTTP RPC endpoint.
func (n *Node) startHTTP(endpoint string, apis []rpc.API, modules []string, cors []string, vhosts []string, timeouts rpc.HTTPTimeouts, wsOrigins []string) error {
	// Short circuit if the HTTP endpoint isn't being exposed
	if endpoint == "" {
		return nil
	}
	// register apis and create handler stack
	srv := rpc.NewServer()
	err := RegisterApisFromWhitelist(apis, modules, srv, false)
	if err != nil {
		return err
	}
	handler := NewHTTPHandlerStack(srv, cors, vhosts)
	// wrap handler in WebSocket handler only if WebSocket port is the same as http rpc
	if n.httpEndpoint == n.wsEndpoint {
		handler = NewWebsocketUpgradeHandler(handler, srv.WebsocketHandler(wsOrigins))
	}
	httpServer, addr, err := StartHTTPEndpoint(endpoint, timeouts, handler)
	if err != nil {
		return err
	}
	n.log.Info("HTTP endpoint opened", "url", fmt.Sprintf("http://%v/", addr),
		"cors", strings.Join(cors, ","),
		"vhosts", strings.Join(vhosts, ","))
	if n.httpEndpoint == n.wsEndpoint {
		n.log.Info("WebSocket endpoint opened", "url", fmt.Sprintf("ws://%v", addr))
	}
	// All listeners booted successfully
	n.httpEndpoint = endpoint
	n.httpListenerAddr = addr
	n.httpServer = httpServer
	n.httpHandler = srv

	return nil
}

// stopHTTP terminates the HTTP RPC endpoint.
func (n *Node) stopHTTP() {
	if n.httpServer != nil {
		// Don't bother imposing a timeout here.
		n.httpServer.Shutdown(context.Background())
		n.log.Info("HTTP endpoint closed", "url", fmt.Sprintf("http://%v/", n.httpListenerAddr))
	}
	if n.httpHandler != nil {
		n.httpHandler.Stop()
		n.httpHandler = nil
	}
}

// startWS initializes and starts the WebSocket RPC endpoint.
func (n *Node) startWS(endpoint string, apis []rpc.API, modules []string, wsOrigins []string, exposeAll bool) error {
	// Short circuit if the WS endpoint isn't being exposed
	if endpoint == "" {
		return nil
	}

	srv := rpc.NewServer()
	handler := srv.WebsocketHandler(wsOrigins)
	err := RegisterApisFromWhitelist(apis, modules, srv, exposeAll)
	if err != nil {
		return err
	}
	httpServer, addr, err := startWSEndpoint(endpoint, handler)
	if err != nil {
		return err
	}
	n.log.Info("WebSocket endpoint opened", "url", fmt.Sprintf("ws://%v", addr))
	// All listeners booted successfully
	n.wsEndpoint = endpoint
	n.wsListenerAddr = addr
	n.wsHTTPServer = httpServer
	n.wsHandler = srv

	return nil
}

// stopWS terminates the WebSocket RPC endpoint.
func (n *Node) stopWS() {
	if n.wsHTTPServer != nil {
		// Don't bother imposing a timeout here.
		n.wsHTTPServer.Shutdown(context.Background())
		n.log.Info("WebSocket endpoint closed", "url", fmt.Sprintf("ws://%v", n.wsListenerAddr))
	}
	if n.wsHandler != nil {
		n.wsHandler.Stop()
		n.wsHandler = nil
	}
}

// Stop terminates a running node along with all it's services. In the node was
// not started, an error is returned.
func (n *Node) Stop() error {
	n.startStopLock.Lock()
	defer n.startStopLock.Unlock()
	//n.lock.Lock()
	//defer n.lock.Unlock()

	// Short circuit if the node's not running
	if n.state != runningState {
		return ErrNodeStopped
	}

	// Terminate the API, services and the p2p server.
	n.stopWS()
	n.stopHTTP()
	n.stopIPC()
	n.rpcAPIs = nil
	failure := &StopError{
		Services: make(map[reflect.Type]error),
	}
	for kind, service := range n.services {
		if err := service.Stop(); err != nil {
			failure.Services[kind] = err
		}
	}
	n.server.Stop()
	n.services = nil
	//n.server = nil

	n.lock.Lock()
	n.state = closedState
	n.lock.Unlock()

	// unblock n.Wait
	//close(n.stop)

	// Remove the keystore if it was created ephemerally.
	var keystoreErr error
	if n.ephemeralKeystore != "" {
		keystoreErr = os.RemoveAll(n.ephemeralKeystore)
	}

	if len(failure.Services) > 0 {
		return failure
	}
	if keystoreErr != nil {
		return keystoreErr
	}
	return nil
}

// Wait blocks the thread until the node is stopped. If the node is not running
// at the time of invocation, the method immediately returns.
func (n *Node) Wait() {
	<-n.stop
}

// Restart terminates a running node and boots up a new one in its place. If the
// node isn't running, an error is returned.
func (n *Node) Restart() error {
	if err := n.Stop(); err != nil {
		return err
	}
	if err := n.Start(); err != nil {
		return err
	}
	return nil
}

// Attach creates an RPC client attached to an in-process API handler.
func (n *Node) Attach() (*rpc.Client, error) {
	n.lock.RLock()
	defer n.lock.RUnlock()

	if n.server == nil {
		return nil, ErrNodeStopped
	}
	return rpc.DialInProc(n.inprocHandler), nil
}

// RPCHandler returns the in-process RPC request handler.
func (n *Node) RPCHandler() (*rpc.Server, error) {
	n.lock.RLock()
	defer n.lock.RUnlock()

	if n.inprocHandler == nil {
		return nil, ErrNodeStopped
	}
	return n.inprocHandler, nil
}

// Server retrieves the currently running P2P network layer. This method is meant
// only to inspect fields of the currently running server, life cycle management
// should be left to this Node entity.
func (n *Node) Server() *p2p.Server {
	//n.lock.RLock()
	//defer n.lock.RUnlock()

	return n.server
}

// Service retrieves a currently running service registered of a specific type.
func (n *Node) Service(service any) error {
	n.lock.RLock()
	defer n.lock.RUnlock()

	// Short circuit if the node's not running
	if n.state != runningState {
		return ErrNodeStopped
	}
	// Otherwise try to find the service to return
	element := reflect.ValueOf(service).Elem()
	if running, ok := n.services[element.Type()]; ok {
		element.Set(reflect.ValueOf(running))
		return nil
	}
	return ErrServiceUnknown
}

// DataDir retrieves the current datadir used by the protocol stack.
// Deprecated: No files should be stored in this directory, use InstanceDir instead.
func (n *Node) DataDir() string {
	return n.config.DataDir
}

// InstanceDir retrieves the instance directory used by the protocol stack.
func (n *Node) InstanceDir() string {
	return n.config.instanceDir()
}

// AccountManager retrieves the account manager used by the protocol stack.
func (n *Node) AccountManager() *accounts.Manager {
	return n.accman
}

// IPCEndpoint retrieves the current IPC endpoint used by the protocol stack.
func (n *Node) IPCEndpoint() string {
	return n.ipcEndpoint
}

// HTTPEndpoint retrieves the current HTTP endpoint used by the protocol stack.
func (n *Node) HTTPEndpoint() string {
	n.lock.Lock()
	defer n.lock.Unlock()

	if n.httpListenerAddr != nil {
		return n.httpListenerAddr.String()
	}
	return n.httpEndpoint
}

// WSEndpoint retrieves the current WS endpoint used by the protocol stack.
func (n *Node) WSEndpoint() string {
	n.lock.Lock()
	defer n.lock.Unlock()

	if n.wsListenerAddr != nil {
		return n.wsListenerAddr.String()
	}
	return n.wsEndpoint
}

// EventMux retrieves the event multiplexer used by all the network services in
// the current protocol stack.
func (n *Node) EventMux() *event.TypeMux {
	return n.eventmux
}

// OpenDatabase opens an existing database with the given name (or creates one if no
// previous can be found) from within the node's instance directory. If the node has no
// data directory, an in-memory database is returned.
func (n *Node) OpenDatabaseWithOptions(name string, opt DatabaseOptions) (ctxcdb.Database, error) {
	//n.lock.Lock()
	//defer n.lock.Unlock()
	if n.state == closedState {
		return nil, ErrNodeStopped
	}
	var db ctxcdb.Database
	var err error
	if n.config.DataDir == "" {
		db, _ = rawdb.Open(memorydb.New(), rawdb.OpenOptions{
			MetricsNamespace: opt.MetricsNamespace,
			ReadOnly:         opt.ReadOnly,
		})
	} else {
		opt.AncientsDirectory = n.ResolveAncient(name, opt.AncientsDirectory)
		db, err = openDatabase(internalOpenOptions{
			directory:       n.ResolvePath(name),
			dbEngine:        n.config.DBEngine,
			DatabaseOptions: opt,
		})
	}
	if err == nil {
		db = n.wrapDatabase(db)
	}
	return db, err
}

// OpenDatabase opens an existing database with the given name (or creates one if no
// previous can be found) from within the node's instance directory.
// If the node has no data directory, an in-memory database is returned.
// Deprecated: use OpenDatabaseWithOptions instead.
func (n *Node) OpenDatabase(name string, cache, handles int, namespace string, readonly bool) (ctxcdb.Database, error) {
	return n.OpenDatabaseWithOptions(name, DatabaseOptions{
		MetricsNamespace: namespace,
		Cache:            cache,
		Handles:          handles,
		ReadOnly:         readonly,
	})
}

// OpenDatabaseWithFreezer opens an existing database with the given name (or
// creates one if no previous can be found) from within the node's data directory.
// If the node has no data directory, an in-memory database is returned.
// Deprecated: use OpenDatabaseWithOptions instead.
func (n *Node) OpenDatabaseWithFreezer(name string, cache, handles int, ancient string, namespace string, readonly bool) (ctxcdb.Database, error) {
	return n.OpenDatabaseWithOptions(name, DatabaseOptions{
		AncientsDirectory: n.ResolveAncient(name, ancient),
		MetricsNamespace:  namespace,
		Cache:             cache,
		Handles:           handles,
		ReadOnly:          readonly,
	})
}

// ResolvePath returns the absolute path of a resource in the instance directory.
func (n *Node) ResolvePath(x string) string {
	return n.config.ResolvePath(x)
}

// apis returns the collection of RPC descriptors this node offers.
func (n *Node) apis() []rpc.API {
	return []rpc.API{
		{
			Namespace: "admin",
			Version:   "1.0",
			Service:   NewPrivateAdminAPI(n),
		}, {
			Namespace: "admin",
			Version:   "1.0",
			Service:   NewPublicAdminAPI(n),
			Public:    true,
		}, {
			Namespace: "debug",
			Version:   "1.0",
			Service:   debug.Handler,
		}, {
			Namespace: "web3",
			Version:   "1.0",
			Service:   NewPublicWeb3API(n),
			Public:    true,
		},
	}
}

// RegisterApisFromWhitelist checks the given modules' availability, generates a whitelist based on the allowed modules,
// and then registers all of the APIs exposed by the services.
func RegisterApisFromWhitelist(apis []rpc.API, modules []string, srv *rpc.Server, exposeAll bool) error {
	if bad, available := checkModuleAvailability(modules, apis); len(bad) > 0 {
		log.Error("Unavailable modules in HTTP API list", "unavailable", bad, "available", available)
	}
	// Generate the whitelist based on the allowed modules
	whitelist := make(map[string]bool)
	for _, module := range modules {
		whitelist[module] = true
	}
	// Register all the APIs exposed by the services
	for _, api := range apis {
		if exposeAll || whitelist[api.Namespace] || (len(whitelist) == 0 && api.Public) {
			if err := srv.RegisterName(api.Namespace, api.Service); err != nil {
				return err
			}
		}
	}
	return nil
}

// ResolveAncient returns the absolute path of the root ancient directory.
func (n *Node) ResolveAncient(name string, ancient string) string {
	switch {
	case ancient == "":
		ancient = filepath.Join(n.ResolvePath(name), "ancient")
	case !filepath.IsAbs(ancient):
		ancient = n.ResolvePath(ancient)
	}
	return ancient
}

// closeTrackingDB wraps the Close method of a database. When the database is closed by the
// service, the wrapper removes it from the node's database map. This ensures that Node
// won't auto-close the database if it is closed by the service that opened it.
type closeTrackingDB struct {
	ctxcdb.Database
	n *Node
}

func (db *closeTrackingDB) Close() error {
	db.n.lock.Lock()
	delete(db.n.databases, db)
	db.n.lock.Unlock()

	db.n.closeDataDir()

	return db.Database.Close()
}

// wrapDatabase ensures the database will be auto-closed when Node is closed.
func (n *Node) wrapDatabase(db ctxcdb.Database) ctxcdb.Database {
	wrapper := &closeTrackingDB{db, n}
	n.databases[wrapper] = struct{}{}
	return wrapper
}

// closeDatabases closes all open databases.
func (n *Node) closeDatabases() (errors []error) {
	for db := range n.databases {
		delete(n.databases, db)
		if err := db.Database.Close(); err != nil {
			errors = append(errors, err)
		}
	}
	return errors
}
