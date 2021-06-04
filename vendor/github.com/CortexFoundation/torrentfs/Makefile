# This Makefile is meant to be used by people that do not usually work
# with Go source code. If you know what GOPATH is then you probably
# don't need to bother with make.

.PHONY: all clean
GOBIN = build/bin
OS = $(shell uname)
ifeq ($(OS), Linux)
endif

ifeq ($(OS), Darwin)
endif

all:
	mkdir -p $(GOBIN)
	go build -v -o $(GOBIN)/torrent cmd/torrent/*.go
	go build -v -o $(GOBIN)/torrent-create cmd/torrent-create/*.go
	go build -v -o $(GOBIN)/torrent-magnet cmd/torrent-magnet/*.go
	go build -v -o $(GOBIN)/seeding cmd/seeding/*.go
clean:
	go clean -cache
	rm -rf $(GOBIN)/*
format:
	find . -name '*.go' -type f -not -path "./vendor*" -not -path "*.git*" -not -path "*/generated/*" | xargs gofmt -w -s
