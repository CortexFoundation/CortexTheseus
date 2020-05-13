BASH_FILE=~/.bashrc

echo "Environment Variable"
echo "    MRT_HOME    : ${MRT_HOME}"
echo "    BASH_FILE   : ${BASH_FILE}"
echo "    PYTHONPATH  : ${PYTHONPATH}"
echo ""

if [[ "x${MRT_HOME}" == "x" ]]; then
  MRT_HOME=`pwd`/cvm/quantization
  echo "export MRT_HOME=${MRT_HOME}" >> ${BASH_FILE}
  echo "export PYTHONPATH=${MRT_HOME}:${PYTHONPATH}" >> ${BASH_FILE}
fi

if [[ "x${TVM_HOME}" == "x" ]]; then
  TVM_HOME=`pwd`
  echo "export TVM_HOME=${TVM_HOME}" >> ${BASH_FILE}
  echo "export PYTHONPATH=${TVM_HOME}/python:${TVM_HOME}/topi/python:${TVM_HOME}/topi/python/topi/testing:${TVM_HOME}/nnvm/python:${TVM_HOME}/vta/python:${PYTHONPATH}" >> ${BASH_FILE}
fi

for p in "$@"; do
  case $p in
    "--pip") ;;
    "--help")
      echo "Usage: "
      echo "    ./install.sh [--help|--pip] "
      echo "    --pip   Install MRT python dependencies."
      exit 0
      ;;
    *) echo "Invalid paramter: $p"; exit -1;;
  esac
done

for p in "$@"; do
  case $p in 
    "--pip")
      echo "Install python dependency packages"
      pip install -r install/requirements.txt \
          -i https://pypi.tuna.tsinghua.edu.cn/simple
      ;;
  esac
done

echo "MRT installed succussfully."
echo "Attention please:"
echo "    source bashfile{${BASH_FILE}} to activate environment."
