# setup libcarma environment
#
# source this file from your ~/.bashrc
#
# relative to <libcarma>/bin/
LIBCARMA=$(cd "$(dirname "$BASH_SOURCE")/.."; pwd)
LIBCARMA_CYTHON=$(cd "$(dirname "$BASH_SOURCE")/../cython"; pwd)

export LIBCARMA=$LIBCARMA
export LIBCARMA_CYTHON=$LIBCARMA_CYTHON
export PYTHONPATH=$PYTHONPATH:$LIBCARMA
export PYTHONPATH=$PAYTHONPATH:$LIBCARMA_CYTHON
export OMP_NUM_THREADS=4
echo "notice: libcarma tools have been set up."