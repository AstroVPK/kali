# setup libcarma environment
#
# source this file from your ~/.bashrc
#
# relative to <libcarma>/bin/
LIBCARMA=$(cd "$(dirname "$BASH_SOURCE")/.."; pwd)
LIBCARMA_CYTHON=$(cd "$(dirname "$BASH_SOURCE")/../python"; pwd)
LIBCARMA_CYTHON_LIB=$(cd "$(dirname "$BASH_SOURCE")/../lib"; pwd)

export LIBCARMA
export PYTHONPATH="$LIBCARMA_CYTHON:$PYTHONPATH"
export PYTHONPATH="$LIBCARMA_CYTHON_LIB:$PYTHONPATH"
echo "notice: libcarma tools have been set up."