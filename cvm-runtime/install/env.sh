CURR_DIR=$(dirname $(realpath $0))
ROOT_DIR=$(dirname ${CURR_DIR})
PROJECT_NAME=$(basename ${ROOT_DIR})

COM=""
LN=$'\n  '

if [[ "${PYTHONPATH}" != *"${PROJECT_NAME}/python"* ]]; then
  read -d '' COM <<EOF
  ${COM}

  PYTHONPATH=${ROOT_DIR}/python:\${PYTHONPATH}
  echo "PYTHONPATH=\${PYTHONPATH}"
EOF
fi

if [[ "${LD_LIBRARY_PATH}" != *"${PROJECT_NAME}/build"* ]]; then
  read -d '' COM <<EOF
  ${COM}

  LD_LIBRARY_PATH=${ROOT_DIR}/build:\${LD_LIBRARY_PATH}
  echo "LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}"
EOF
fi

if [[ ${COM} != "" ]]; then
  cat <<EOF

According to bash limitation, we cannot add python & link library 
  environment via scripts, and then we supply the below commands to 
  help to setup the project, copy and execute it in terminal please:

\`
  ${COM}
\`

EOF
fi

echo "Done."

# compile the cython module
# echo "Compile the cython setup"
# cd python
# python3 setup.py build_ext -i
# cd ..
