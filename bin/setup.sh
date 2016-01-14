# setup libcarma environment
#
# source this file from your ~/.bashrc
#
# relative to <libcarma>/bin/
LIBCARMA=$(cd "$(dirname "$BASH_SOURCE")/.."; pwd)

export PYTHONPATH=$LIBCARMA
export OMP_NUM_THREADS=4
echo "notice: libcarma tools have been set up."