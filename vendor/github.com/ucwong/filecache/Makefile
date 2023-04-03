# This Makefile is meant to be used by people that do not usually work
# with Go source code. If you know what GOPATH is then you probably
# don't need to bother with make.

.PHONY: all clean test
GOBIN = build/bin
OS = $(shell uname)
ifeq ($(OS), Linux)
endif

ifeq ($(OS), Darwin)
endif

format:
	find . -name '*.go' -type f -not -path "./vendor*" -not -path "*.git*" -not -path "*/generated/*" | xargs gofmt -w -s
test:
	go test ./... -v -race -cpu=1,2 -coverprofile=coverage.txt -covermode=atomic -benchmem -bench .
clean:
	rm -rf coverage.txt
	go clean -cache
