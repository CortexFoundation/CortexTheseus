FROM golang:1.16-alpine as builder

RUN apk add --no-cache make gcc g++ musl-dev linux-headers git python3 cmake libexecinfo-dev py3-pip \
        libexecinfo \
	&& mkdir /src \
    	&& cd /src \
    	&& git clone --depth 1 --recursive -b v0.81 https://github.com/dmlc/xgboost \
    	&& ln -s locale.h /usr/include/xlocale.h \
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
RUN cd /src && git clone https://github.com/CortexFoundation/CortexTheseus.git \
  && cd CortexTheseus \
  && git checkout 4a71927373952172aa76d235b5e479988388ea14 \
  && make

#RUN cd /src/CortexTheseus && make -j8

FROM alpine:latest

RUN apk add --no-cache ca-certificates
COPY --from=builder /src/CortexTheseus/build/bin/cortex /usr/local/bin/

#RUN rm -rf CortexTheseus

EXPOSE 8545 8546 8547 40404 40404/udp 40401 40401/udp
ENTRYPOINT ["cortex"]
