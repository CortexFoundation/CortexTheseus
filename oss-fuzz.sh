#/bin/bash -eu
# Copyright 2020 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

# This file is for integration with Google OSS-Fuzz.
# The following ENV variables are available when executing on OSS-fuzz:
#
# /out/         $OUT    Directory to store build artifacts (fuzz targets, dictionaries, options files, seed corpus archives).
# /src/         $SRC    Directory to checkout source files.
# /work/        $WORK   Directory to store intermediate files.
#
# $CC, $CXX, $CCC       The C and C++ compiler binaries.
# $CFLAGS, $CXXFLAGS    C and C++ compiler flags.
# $LIB_FUZZING_ENGINE   C++ compiler argument to link fuzz target against the prebuilt engine library (e.g. libFuzzer).

# This sets the -coverpgk for the coverage report when the corpus is executed through go test
coverpkg="github.com/CortexFoundation/CortexTheseus/..."

function coverbuild {
  path=$1
  function=$2
  fuzzer=$3
  tags=""

  if [[ $#  -eq 4 ]]; then
    tags="-tags $4"
  fi
  cd $path
  fuzzed_package=`pwd | rev | cut -d'/' -f 1 | rev`
  cp $GOPATH/ossfuzz_coverage_runner.go ./"${function,,}"_test.go
  sed -i -e 's/FuzzFunction/'$function'/' ./"${function,,}"_test.go
  sed -i -e 's/mypackagebeingfuzzed/'$fuzzed_package'/' ./"${function,,}"_test.go
  sed -i -e 's/TestFuzzCorpus/Test'$function'Corpus/' ./"${function,,}"_test.go

cat << DOG > $OUT/$fuzzer
#/bin/sh

  cd $OUT/$path
  go test -run Test${function}Corpus -v $tags -coverprofile \$1 -coverpkg $coverpkg

DOG

  chmod +x $OUT/$fuzzer
  #echo "Built script $OUT/$fuzzer"
  #cat $OUT/$fuzzer
  cd -
}

function compile_fuzzer {
  # Inputs:
  # $1: The package to fuzz, within CortexTheseus
  # $2: The name of the fuzzing function
  # $3: The name to give to the final fuzzing-binary

  path=$GOPATH/src/github.com/CortexFoundation/CortexTheseus/$1
  func=$2
  fuzzer=$3

  echo "Building $fuzzer"

  # Do a coverage-build or a regular build
  if [[ $SANITIZER = *coverage* ]]; then
    coverbuild $path $func $fuzzer $coverpkg
  else
    (cd $path && \
        go-fuzz -func $func -o $WORK/$fuzzer.a . && \
        $CXX $CXXFLAGS $LIB_FUZZING_ENGINE $WORK/$fuzzer.a -o $OUT/$fuzzer)
  fi

  ## Check if there exists a seed corpus file
  corpusfile="${path}/testdata/${fuzzer}_seed_corpus.zip"
  if [ -f $corpusfile ]
  then
    cp $corpusfile $OUT/
    echo "Found seed corpus: $corpusfile"
  fi
}

go install github.com/holiman/gofuzz-shim@latest
repo=$GOPATH/src/github.com/ethereum/go-ethereum
compile_fuzzer github.com/ethereum/go-ethereum/accounts/abi \
  FuzzABI fuzzAbi \
  $repo/accounts/abi/abifuzzer_test.go

compile_fuzzer github.com/ethereum/go-ethereum/common/bitutil \
  FuzzEncoder fuzzBitutilEncoder \
  $repo/common/bitutil/compress_test.go

compile_fuzzer github.com/ethereum/go-ethereum/common/bitutil \
  FuzzDecoder fuzzBitutilDecoder \
  $repo/common/bitutil/compress_test.go

compile_fuzzer github.com/ethereum/go-ethereum/core/vm/runtime \
  FuzzVmRuntime fuzzVmRuntime\
  $repo/core/vm/runtime/runtime_fuzz_test.go

compile_fuzzer github.com/ethereum/go-ethereum/core/vm \
  FuzzPrecompiledContracts fuzzPrecompiledContracts\
  $repo/core/vm/contracts_fuzz_test.go,$repo/core/vm/contracts_test.go

compile_fuzzer github.com/ethereum/go-ethereum/core/types \
  FuzzRLP fuzzRlp \
  $repo/core/types/rlp_fuzzer_test.go

compile_fuzzer github.com/ethereum/go-ethereum/crypto/blake2b \
  Fuzz fuzzBlake2b \
  $repo/crypto/blake2b/blake2b_f_fuzz_test.go

compile_fuzzer github.com/ethereum/go-ethereum/accounts/keystore \
  FuzzPassword fuzzKeystore \
  $repo/accounts/keystore/keystore_fuzzing_test.go

pkg=$repo/trie/
compile_fuzzer github.com/ethereum/go-ethereum/trie \
  FuzzTrie fuzzTrie \
  $pkg/trie_test.go,$pkg/database_test.go,$pkg/tracer_test.go,$pkg/proof_test.go,$pkg/iterator_test.go,$pkg/sync_test.go

compile_fuzzer github.com/ethereum/go-ethereum/trie \
  FuzzStackTrie fuzzStackTrie \
  $pkg/stacktrie_fuzzer_test.go,$pkg/iterator_test.go,$pkg/trie_test.go,$pkg/database_test.go,$pkg/tracer_test.go,$pkg/proof_test.go,$pkg/sync_test.go

#compile_fuzzer tests/fuzzers/snap  FuzzARange fuzz_account_range
compile_fuzzer github.com/ethereum/go-ethereum/eth/protocols/snap \
  FuzzARange fuzz_account_range \
  $repo/eth/protocols/snap/handler_fuzzing_test.go

compile_fuzzer github.com/ethereum/go-ethereum/eth/protocols/snap \
  FuzzSRange fuzz_storage_range \
  $repo/eth/protocols/snap/handler_fuzzing_test.go

compile_fuzzer github.com/ethereum/go-ethereum/eth/protocols/snap \
  FuzzByteCodes fuzz_byte_codes \
  $repo/eth/protocols/snap/handler_fuzzing_test.go

compile_fuzzer github.com/ethereum/go-ethereum/eth/protocols/snap \
  FuzzTrieNodes fuzz_trie_nodes\
  $repo/eth/protocols/snap/handler_fuzzing_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bn256 \
  FuzzAdd fuzzBn256Add\
  $repo/tests/fuzzers/bn256/bn256_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bn256 \
  FuzzMul fuzzBn256Mul \
  $repo/tests/fuzzers/bn256/bn256_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bn256 \
  FuzzPair fuzzBn256Pair \
  $repo/tests/fuzzers/bn256/bn256_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/txfetcher \
  Fuzz fuzzTxfetcher \
  $repo/tests/fuzzers/txfetcher/txfetcher_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzG1Add fuzz_g1_add\
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzG1Mul fuzz_g1_mul\
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzG1MultiExp fuzz_g1_multiexp \
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzG2Add fuzz_g2_add \
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzG2Mul fuzz_g2_mul\
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzG2MultiExp fuzz_g2_multiexp \
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzPairing fuzz_pairing \
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzMapG1 fuzz_map_g1\
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzMapG2 fuzz_map_g2 \
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzCrossG1Add fuzz_cross_g1_add \
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzCrossG1MultiExp fuzz_cross_g1_multiexp \
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzCrossG2Add fuzz_cross_g2_add \
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzCrossPairing fuzz_cross_pairing\
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzG1SubgroupChecks fuzz_g1_subgroup_checks\
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/bls12381 \
  FuzzG2SubgroupChecks fuzz_g2_subgroup_checks\
  $repo/tests/fuzzers/bls12381/bls12381_test.go

compile_fuzzer github.com/ethereum/go-ethereum/tests/fuzzers/secp256k1 \
  Fuzz fuzzSecp256k1\
  $repo/tests/fuzzers/secp256k1/secp_test.go


#compile_fuzzer tests/fuzzers/vflux      FuzzClientPool fuzzClientPool
#compile_fuzzer tests/fuzzers/difficulty Fuzz fuzzDifficulty
#compile_fuzzer tests/fuzzers/les        Fuzz fuzzLes
