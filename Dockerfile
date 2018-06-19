FROM python:3.6.5-slim-stretch

WORKDIR /app

ENV LD_LIBRARY_PATH /usr/local/lib

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y cmake git libgtk2.0-dev libblas-dev liblapack-dev \
  && rm -rf /var/lib/apt/lists/*

# install ngt
RUN git clone https://github.com/yahoojapan/NGT.git \
  && cd NGT \
  && git checkout v1.3.3 \
  && mkdir build && cd build \
  && cmake .. \
  && make \
  && make install \
  && cd ../python \
  && pip install --upgrade pip==9.0.3 \
  && pip install pybind11 \
  && python setup.py sdist \
  && pip install dist/ngt-1.1.0.tar.gz

# install hnsw
RUN git clone https://github.com/nmslib/hnsw.git \
  && cd hnsw/python_bindings \
  && pip install numpy \
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

RUN pip install imgdupes

ENTRYPOINT ["imgdupes"]
