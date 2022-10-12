.PHONY: all

all: test cover

test:
	go test ./... -v -race -cpu=1,2,4 -coverprofile=coverage.txt -covermode=atomic -benchmem -bench .

cover:
	go tool cover -html=coverage.txt -o coverage.html

clean:
	go clean -cache
	find . -name '.golang-kv' | xargs rm -rf
	rm -rf coverage.html
	rm -rf coverage.txt
