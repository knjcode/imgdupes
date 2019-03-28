FROM python:3.6.8-slim-stretch

WORKDIR /app

ENV LD_LIBRARY_PATH /usr/local/lib

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y cmake git libgtk2.0-dev libblas-dev liblapack-dev \
  && rm -rf /var/lib/apt/lists/*

# install hnsw
RUN git clone https://github.com/nmslib/hnsw.git \
  && cd hnsw/python_bindings \
  && pip install numpy pybind11 \
  && python setup.py install

# install faiss
RUN git clone https://github.com/facebookresearch/faiss.git \
  && cd faiss \
  && ./configure \
  && make \
  && make install \
  && make py \
  && cd python \
  && python setup.py install

ARG VERSION
COPY dist/imgdupes-${VERSION}.tar.gz .
RUN pip install imgdupes-${VERSION}.tar.gz

ENTRYPOINT ["imgdupes"]
