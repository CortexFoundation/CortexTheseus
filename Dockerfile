#FROM golang:1.16-alpine as builder
FROM ubuntu:18.04 as golang-builder
RUN apt-get update && apt-get install -y curl make gcc g++ git python3 cmake supervisor
ENV GOLANG_VERSION 1.16.4
ENV GOLANG_DOWNLOAD_SHA256 7154e88f5a8047aad4b80ebace58a059e36e7e2e4eb3b383127a28c711b4ff59
ENV GOLANG_DOWNLOAD_URL https://golang.org/dl/go$GOLANG_VERSION.linux-amd64.tar.gz

RUN curl -fsSL "$GOLANG_DOWNLOAD_URL" -o golang.tar.gz \
  && echo "$GOLANG_DOWNLOAD_SHA256  golang.tar.gz" | sha256sum -c - \
  && tar -C /usr/local -xzf golang.tar.gz \
  && rm golang.tar.gz

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"
RUN mkdir -p /work/src

run mkdir -p /work/bin
RUN cd /work/src && git clone https://github.com/CortexFoundation/CortexTheseus.git \
  && cd CortexTheseus \
  && git checkout d1ebbd8c6e5f6d83a89fe302da2f16acfc608a80 \
  && make

#RUN cd /src/CortexTheseus && make -j8

RUN cp -r /work/src/CortexTheseus/build/bin/cortex /work/bin/
RUN cp -r /work/src/CortexTheseus/plugins /work/bin

RUN rm -rf /work/src/CortexTheseus

WORKDIR /work/bin

RUN ls /work/bin

run cp /work/src/CortexTheseus/node.conf /etc/supervisor/conf.d/
run cat /etc/supervisor/conf.d/node.conf
#"update and restart supervisorctl"
run service supervisor start
cmd supervisorctl reread
cmd supervisorctl update
cmd supervisorctl restart all

EXPOSE 8545 8546 8547 40404 40404/udp 40401 40401/udp
#ENTRYPOINT ["cortex"]
#CMD ["/usr/local/bin/cortex", "run"]
