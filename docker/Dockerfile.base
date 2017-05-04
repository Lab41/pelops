FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

MAINTAINER Lab41 <info@lab41.org>

RUN apt-get update && \
    apt-get install -y \
    bzip2 \
    ca-certificates \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget

#Configure environment
ENV CONDA_DIR=/opt/conda \
    # 4.2.12 is the last version with Python3.5, which we need
    MINICONDA_SCRIPT=Miniconda3-4.2.12-Linux-x86_64.sh \
    MINICONDA_SHA=c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a

# Install conda
RUN cd /tmp && \
    mkdir -p $CONDA_DIR && \
    wget --quiet https://repo.continuum.io/miniconda/${MINICONDA_SCRIPT} && \
    echo "${MINICONDA_SHA} ${MINICONDA_SCRIPT}" | sha256sum -c - && \
    /bin/bash ${MINICONDA_SCRIPT} -f -b -p $CONDA_DIR && \
    rm ${MINICONDA_SCRIPT}

RUN $CONDA_DIR/bin/conda install --quiet --yes \
    'conda-build=2.1.*' \
    'cython=0.24*' \
    'h5py=2.6*' \
    'hdfs3=0.1.*' \
    'libhdfs3=2.2.*' \
    'numpy=1.11*' \
    'pillow=3.4*' \
    'pytest=3.0.*' \
    'python=3.5.*' \
    'scikit-image=0.12*' \
    'scikit-learn=0.18*' \
    && $CONDA_DIR/bin/conda clean -tipsy

RUN $CONDA_DIR/bin/conda update pip --quiet --yes

# Install Python packages
ENV TENSORFLOW_VERSION=0.12.* \
    KERAS_VERSION=2ad3544b017fe9c0d7a25ef0640baa52281372b5
RUN $CONDA_DIR/bin/pip install git+git://github.com/fchollet/keras.git@${KERAS_VERSION} \
    tensorflow==${TENSORFLOW_VERSION} \
    imageio

ENV INDOCKERCONTAINER 1

ADD . /pelops_root
WORKDIR /pelops_root
ENV PYTHONPATH=/pelops_root/pelops:$PYTHONPATH \
    PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:$CONDA_DIR/bin:$PATH

# install dependencies of plugins for pelops
RUN for file in $(find . -name "requirements.txt"); \
    do \
        $CONDA_DIR/bin/pip install -r $file; \
    done
