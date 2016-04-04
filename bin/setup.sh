# setup libcarma environment
#
# source this file from your ~/.bashrc
#
# relative to <libcarma>/bin/
LIBCARMA=$(cd "$(dirname "$BASH_SOURCE")/.."; pwd)
LIBCARMA_CYTHON=$(cd "$(dirname "$BASH_SOURCE")/../cython/python"; pwd)
LIBCARMA_CYTHON_LIB=$(cd "$(dirname "$BASH_SOURCE")/../cython/lib"; pwd)

export LIBCARMA="$LIBCARMA"
export LIBCARMA_CYTHON="$LIBCARMA_CYTHON"
export LIBCARMA_CYTHON_LIB="$LIBCARMA_CYTHON_LIB"
export PYTHONPATH="$LIBCARMA:$PYTHONPATH"
export PYTHONPATH="$LIBCARMA_CYTHON:$PYTHONPATH"
export PYTHONPATH="$LIBCARMA_CYTHON_LIB:$PYTHONPATH"
export OMP_NUM_THREADS=4
echo "notice: libcarma tools have been set up."