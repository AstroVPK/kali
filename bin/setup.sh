# setup libcarma environment
#
# source this file from your ~/.bashrc
#
# relative to <libcarma>/bin/
LIBCARMA=$(cd "$(dirname "$BASH_SOURCE")/.."; pwd)
LIBCARMA_CFFI=$(cd "$(dirname "$BASH_SOURCE")/../cffi"; pwd)
LIBCARMA_CYTHON=$(cd "$(dirname "$BASH_SOURCE")/../python"; pwd)
LIBCARMA_CYTHON_LIB=$(cd "$(dirname "$BASH_SOURCE")/../lib"; pwd)

export LIBCARMA="$LIBCARMA"
export LIBCARMA_CFFI="$LIBCARMA_CFFI"
export LIBCARMA_CYTHON="$LIBCARMA_CYTHON"
export LIBCARMA_CYTHON_LIB="$LIBCARMA_CYTHON_LIB"
export PYTHONPATH="$LIBCARMA_CFFI:$PYTHONPATH"
export PYTHONPATH="$LIBCARMA_CYTHON:$PYTHONPATH"
export PYTHONPATH="$LIBCARMA_CYTHON_LIB:$PYTHONPATH"
export OMP_NUM_THREADS=4
echo "notice: libcarma tools have been set up."