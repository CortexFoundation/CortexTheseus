FROM golang:1.14-alpine as builder

RUN apk add --no-cache make gcc musl-dev linux-headers git

ADD . /CortexTheseus
RUN cd /CortexTheseus && make

FROM alpine:latest

RUN apk add --no-cache ca-certificates
COPY --from=builder /CortexTheseus/build/bin/cortex /usr/local/bin/

EXPOSE 8545 8546 8547 40404 40404/udp 40401 40401/udp
ENTRYPOINT ["cortex"]
