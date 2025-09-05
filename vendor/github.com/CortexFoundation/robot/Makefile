.PHONY: fmt lint test bench build docker

fmt:
ifeq ($(shell test -e go.mod && echo yes),yes)
	gofmt -s -w .
else
	prettier --write .
endif

lint:
	golangci-lint run ./...

test:
	go test ./...

bench:
	go test -bench . -benchmem ./...

build:
	go build -o bin/app ./cmd/...

docker:
	docker build -t myorg/robot:latest .

