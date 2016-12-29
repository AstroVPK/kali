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
    g++

# Install MKL dependency packages
    RUN apt-get update && \
    apt-get install -y man gfortran

# Install MKL
RUN cd /tmp && \
# Download MKL install package
    wget -q https://registrationcenter.intel.com/en/products/download/2884/l_mkl_2017.1.132.tgz && \
# Install MKL
tar -xzf l_mkl_2017.1.132.tgz && cd l_mkl_2017.1.132.tgz && \
sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' silent.cfg && \
sed -i 's/ACTIVATION_TYPE=exist_lic/ACTIVATION_TYPE=trial_lic/g' silent.cfg && \
./install.sh -s silent.cfg && \
# Clean up
cd .. && rm -rf *
RUN echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && ldconfig && \
    echo ". /opt/intel/bin/compilervars.sh intel64" >> /etc/bash.bashrc

# Install Anaconda2
FROM qnib/u-syslog

ENV ANACONDA_VER=4.0.0 \
    ANACONDA_PY_VER=2
RUN apt-get install -y wget bzip2 \
 && echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh \
 && wget --quiet https://repo.continuum.io/archive/Anaconda${ANACONDA_PY_VER}-${ANACONDA_VER}-Linux-x86_64.sh \
 &&  /bin/bash /Anaconda${ANACONDA_PY_VER}-${ANACONDA_VER}-Linux-x86_64.sh -b -p /opt/conda \
 && rm /Anaconda${ANACONDA_PY_VER}-${ANACONDA_VER}-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH \
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
    LANG=C.UTF-8

# Install kali
RUN python setup.py build_ext
CMD source ./bin/setup.sh
