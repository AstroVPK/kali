# Start with Ubuntu base image
FROM ubuntu:16.04

MAINTAINER AstroVPK <vishal.kasliwal@gmail.com>

# For SSH access and port redirection
ENV ROOTPASSWORD sample

# Turn off prompts during installations
ENV DEBIAN_FRONTEND noninteractive
RUN echo "debconf shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections
RUN echo "debconf shared/accepted-oracle-license-v1-1 seen true" | debconf-set-selections

# Update packages
RUN apt-get -y update

# Install system tools / libraries
RUN apt-get -y install python3-software-properties \
    software-properties-common \
    bzip2 \
    ssh \
    net-tools \
    vim \
    curl \
    expect \
    git \
    nano \
    wget \
    build-essential \
    dialog \
    make \
    build-essential \
    checkinstall \
    bridge-utils \
    virt-viewer \
    python-pip \
    python-setuptools \
    python-dev \
    g++ \
    cmake \
    man \
    gfortran \
    cpio

# Install MKL
WORKDIR /tmp
RUN wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15816/l_mkl_2019.5.281_online.tgz 
RUN tar -xzf l_mkl_2019.5.281_online.tgz
WORKDIR /tmp/l_mkl_2019.5.281_online
RUN sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
sed -i 's/ACTIVATION_TYPE=exist_lic/ACTIVATION_TYPE=trial_lic/g' silent.cfg && \
./install.sh -s silent.cfg && \
# Clean up
cd .. && rm -rf *

# CONDA
WORKDIR /tmp
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH=/opt/conda/bin:$PATH \
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
    LANG=C.UTF-8

# NLOPT
RUN wget https://github.com/stevengj/nlopt/archive/v2.6.1.tar.gz  -O nlopt.tar.gz 
RUN tar -xvzf nlopt.tar.gz && cd nlopt-2.6.1/ && cmake . && make && make install && ldconfig && cd ..

# Kali Requirements
RUN conda install cython future pytest numpy matplotlib scipy jupyter
RUN pip install gatspy multi_key_dict fitsio==0.9.8 acor brewer2mpl sklearn zmq psutil astropy

# Install kali
WORKDIR /home
COPY . . 
WORKDIR /home/kali/
RUN COMPILERVARS_ARCHITECTURE=intel64 . /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh && \
    . /opt/intel/mkl/bin/mklvars.sh intel64 && python setup.py build_ext

# Setup
SHELL ["/bin/bash", "-c"]
RUN echo "source /home/kali/bin/setup.sh" >> ~/.bashrc
RUN echo "DISPLAY=:0.0" >> ~/.bashrc
ENTRYPOINT [ "/bin/bash", "-c" ]
