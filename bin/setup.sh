# setup libcarma environment
#
# source this file from your ~/.bashrc
#
# relative to <libcarma>/bin/
LIBCARMA_CYTHON=$(cd "$(dirname "$BASH_SOURCE")/../python"; pwd)
LIBCARMA_CYTHON_LIB=$(cd "$(dirname "$BASH_SOURCE")/../lib"; pwd)

export PYTHONPATH="$LIBCARMA_CYTHON:$PYTHONPATH"
export PYTHONPATH="$LIBCARMA_CYTHON_LIB:$PYTHONPATH"
export OMP_NUM_THREADS=4
echo "notice: libcarma tools have been set up."