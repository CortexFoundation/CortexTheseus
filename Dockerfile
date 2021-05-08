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
RUN mkdir /src
#RUN apk add --no-cache make gcc g++ musl-dev linux-headers git python3 cmake libexecinfo-dev py3-pip \
#        libexecinfo \
#	&& mkdir /src \
#    	&& cd /src \
#    	&& git clone --depth 1 --recursive -b v0.81 https://github.com/dmlc/xgboost \
#    	&& ln -s locale.h /usr/include/xlocale.h \
    # update dmlc-core to have better backtrace detection logic
    # so that musl in alpine can work
    # https://github.com/dmlc/dmlc-core/pull/487
    #&& cd /src/xgboost/dmlc-core; git checkout master && git pull \ 
    #&& cd /src/xgboost; mkdir build; cd build; cmake ..; make -j4 \
    #&& cd /src/xgboost/python-package \
    #&& pip install numpy~=1.15.4 \
    #&& pip install scipy~=1.2.0 \
    #&& pip install pandas~=0.23.4 \
    #&& pip install scikit-learn~=0.20.2 \    
    #&& python3 setup.py install \

#ADD . /CortexTheseus
run mkdir -p /work/bin
RUN cd /src && git clone https://github.com/CortexFoundation/CortexTheseus.git \
  && cd CortexTheseus \
  && git checkout 4a71927373952172aa76d235b5e479988388ea14 \
  && make

#RUN cd /src/CortexTheseus && make -j8

RUN cp -r /src/CortexTheseus/build/bin/cortex /work/bin/
RUN cp -r /src/CortexTheseus/plugins /work/bin

RUN rm -rf /src/CortexTheseus

add node.conf /etc/supervisor/conf.d/
run cat /etc/supervisor/conf.d/node.conf
#"update and restart supervisorctl"
run service supervisor start
cmd supervisorctl reread
cmd supervisorctl update
cmd supervisorctl restart all

EXPOSE 8545 8546 8547 40404 40404/udp 40401 40401/udp
#ENTRYPOINT ["cortex"]
#CMD ["/usr/local/bin/cortex", "run"]
