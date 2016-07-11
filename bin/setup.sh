# setup kali environment
#
# source this file from your ~/.bashrc
#
# relative to <kali>/bin/
KALI=$(cd "$(dirname "$BASH_SOURCE")/.."; pwd)
KALI_CYTHON=$(cd "$(dirname "$BASH_SOURCE")/../python"; pwd)
KALI_CYTHON_LIB=$(cd "$(dirname "$BASH_SOURCE")/../lib"; pwd)

export KALI
export PYTHONPATH="$KALI_CYTHON:$PYTHONPATH"
export PYTHONPATH="$KALI_CYTHON_LIB:$PYTHONPATH"
echo "notice: kali tools have been set up."