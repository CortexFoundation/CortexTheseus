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

RUN mkdir -p /work/bin
RUN cd /work/src && git clone https://github.com/CortexFoundation/CortexTheseus.git \
  && cd CortexTheseus \
  && git checkout 94accbb786a1d95399450c21eb59da6cabe64638 \
  && make

RUN cp -r /work/src/CortexTheseus/build/bin/cortex /work/bin/
RUN cp -r /work/src/CortexTheseus/plugins /work/bin

WORKDIR /work/bin

RUN ls /work/bin

RUN cp /work/src/CortexTheseus/node.conf /etc/supervisor/conf.d/

RUN rm -rf /work/src/CortexTheseus

CMD supervisord -n -c /etc/supervisor/supervisord.conf

EXPOSE 8545 8546 8547 40404 40404/udp 40401 40401/udp
