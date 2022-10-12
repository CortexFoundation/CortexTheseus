.PHONY: all test cover

all: test cover

test:
	go test -v -race -cpu=1,2,4 -coverprofile=coverage.txt -covermode=atomic -benchmem -bench .

cover:
	go tool cover -html=coverage.txt -o coverage.html
