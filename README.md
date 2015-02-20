## Ethereum Go

Ethereum Go Client © 2014 Jeffrey Wilcke.

          | Linux   | OSX | Windows | Tests
----------|---------|-----|---------|------
develop   | [![Build+Status](http://build.ethdev.com/buildstatusimage?builder=Linux%20Go%20develop%20branch)](http://build.ethdev.com/builders/Linux%20Go%20develop%20branch/builds/-1) | [![Build+Status](http://build.ethdev.com/buildstatusimage?builder=Linux%20Go%20develop%20branch)](http://build.ethdev.com/builders/OSX%20Go%20develop%20branch/builds/-1) | N/A | [![Buildr+Status](https://travis-ci.org/ethereum/go-ethereum.svg?branch=develop)](https://travis-ci.org/ethereum/go-ethereum)
master    | [![Build+Status](http://build.ethdev.com/buildstatusimage?builder=Linux%20Go%20master%20branch)](http://build.ethdev.com/builders/Linux%20Go%20master%20branch/builds/-1) | [![Build+Status](http://build.ethdev.com/buildstatusimage?builder=OSX%20Go%20master%20branch)](http://build.ethdev.com/builders/OSX%20Go%20master%20branch/builds/-1) | N/A | [![Buildr+Status](https://travis-ci.org/ethereum/go-ethereum.svg?branch=master)](https://travis-ci.org/ethereum/go-ethereum)

[![Bugs](https://badge.waffle.io/ethereum/go-ethereum.png?label=bug&title=Bugs)](https://waffle.io/ethereum/go-ethereum)
[![Stories in Ready](https://badge.waffle.io/ethereum/go-ethereum.png?label=ready&title=Ready)](https://waffle.io/ethereum/go-ethereum)
[![Stories in Progress](https://badge.waffle.io/ethereum/go-ethereum.svg?label=in%20progress&title=In Progress)](http://waffle.io/ethereum/go-ethereum)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ethereum/go-ethereum?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)


Build
=====

Mist (GUI):

`go get github.com/ethereum/go-ethereum/cmd/mist`

Ethereum (CLI):

`go get github.com/ethereum/go-ethereum/cmd/ethereum`

For further, detailed, build instruction please see the [Wiki](https://github.com/ethereum/go-ethereum/wiki/Building-Ethereum(Go))

Automated (dev) builds
======================

* [[Docker](https://registry.hub.docker.com/u/ethereum/client-go/)]
* [[OS X](http://build.ethdev.com/builds/OSX%20Go%20develop%20branch/Mist-OSX-latest.dmg)]
* [Windows] Coming soon&trade;
* [Linux] Coming soon&trade;

Binaries
========

Go Ethereum comes with several binaries found in
[cmd](https://github.com/ethereum/go-ethereum/tree/master/cmd):

* `mist` Official Ethereum Browser
* `ethereum` Ethereum CLI
* `ethtest` test tool which runs with the [tests](https://github.com/ethereum/testes) suit: 
  `cat file | ethtest`.
* `evm` is a generic Ethereum Virtual Machine: `evm -code 60ff60ff -gas
  10000 -price 0 -dump`. See `-h` for a detailed description.
* `rlpdump` converts a rlp stream to `interface{}`.
* `peerserver` simple P2P (noi-ethereum) peer server.
* `disasm` disassembles EVM code: `echo "6001" | disasm`

General command line options
============================

```
== Shared between ethereum and Mist ==

= Settings
-id      Set the custom identifier of the client (shows up on other clients)
-port    Port on which the server will accept incomming connections
-upnp    Enable UPnP
-maxpeer Desired amount of peers
-rpc     Start JSON RPC
-dir     Data directory used to store configs and databases

= Utility 
-h         This
-import    Import a private key
-genaddr   Generates a new address and private key (destructive action)
-dump      Dump a specific state of a block to stdout given the -number or -hash
-difftool  Supress all output and prints VM output to stdout
-diff      vm=only vm output, all=all output including state storage

Ethereum only
ethereum [options] [filename]
-js        Start the JavaScript REPL
filename   Load the given file and interpret as JavaScript
-m       Start mining blocks

== Mist only ==

-asset_path    absolute path to GUI assets directory
```

Contribution
============

If you'd like to contribute to Ethereum please fork, fix, commit and
send a pull request. Commits who do not comply with the coding standards
are ignored (use gofmt!). If you send pull requests make absolute sure that you
commit on the `develop` branch and that you do not merge to master.
Commits that are directly based on master are simply ignored.

For dependency management, we use [godep](https://github.com/tools/godep). After installing with `go get github.com/tools/godep`, run `godep restore` to ensure that changes to other repositories do not break the build. To update a dependency version (for example, to include a new upstream fix), run `go get -u <foo/bar>` then `godep update <foo/...>`. To track a new dependency, add it to the project as normal than run `godep save ./...`. Changes to the Godeps folder should be manually verified then commited.

To make life easier try [git flow](http://nvie.com/posts/a-successful-git-branching-model/) it sets
this all up and streamlines your work flow.

Coding standards
================

Sources should be formatted according to the [Go Formatting
Style](http://golang.org/doc/effective_go.html#formatting).

Unless structs fields are supposed to be directly accesible, provide
Getters and hide the fields through Go's exporting facility.

When you comment put meaningful comments. Describe in detail what you
want to achieve.

*wrong*

```go
// Check if the value at x is greater than y
if x > y {
    // It's greater!
}
```

Everyone reading the source probably know what you wanted to achieve
with above code. Those are **not** meaningful comments.

While the project isn't 100% tested I want you to write tests non the
less. I haven't got time to evaluate everyone's code in detail so I
expect you to write tests for me so I don't have to test your code
manually. (If you want to contribute by just writing tests that's fine
too!)


