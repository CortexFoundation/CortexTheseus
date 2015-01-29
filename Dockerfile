FROM ubuntu:14.04

## Environment setup
ENV HOME /root
ENV GOPATH /root/go
ENV PATH /go/bin:/root/go/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games

RUN mkdir -p /root/go
ENV DEBIAN_FRONTEND noninteractive

## Install base dependencies
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git mercurial build-essential software-properties-common pkg-config libgmp3-dev libreadline6-dev libpcre3-dev libpcre++-dev

## Build and install Go
RUN git clone https://go.googlesource.com/go golang
RUN cd golang && git checkout go1.4.1
RUN cd golang/src && ./all.bash && go version

## Install GUI dependencies
RUN add-apt-repository ppa:ubuntu-sdk-team/ppa -y
RUN apt-get update -y
RUN apt-get install -y qtbase5-private-dev qtdeclarative5-private-dev libqt5opengl5-dev

## Fetch and install serpent-go
RUN go get -v -d github.com/ethereum/serpent-go
WORKDIR $GOPATH/src/github.com/ethereum/serpent-go
# RUN git checkout master
RUN git submodule update --init
RUN go install -v

# Fetch and install go-ethereum
RUN go get -v -d github.com/ethereum/go-ethereum/...
WORKDIR $GOPATH/src/github.com/ethereum/go-ethereum
# RUN git checkout develop
RUN ETH_DEPS=$(go list -f '{{.Imports}} {{.TestImports}} {{.XTestImports}}' github.com/ethereum/go-ethereum/... | sed -e 's/\[//g' | sed -e 's/\]//g' | sed -e 's/C //g'); if [ "$ETH_DEPS" ]; then go get $ETH_DEPS; fi
RUN go install -v ./cmd/ethereum

# Run JSON RPC
ENTRYPOINT ["ethereum", "-rpc=true", "-rpcport=8080"]
EXPOSE 8080
