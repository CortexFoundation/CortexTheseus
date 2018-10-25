#!/bin/bash

TYPE=$1
shift

DEFAULT_DES_DIR=./root

_print_start() {
	echo
	echo "################ $1 #################"
	echo 
	echo 
}

_print_end() {
	echo "#############################################"
}

_binary_collect() {
	PACKAGE=$1
	P_PATH=${PACKAGE%/*}
	P_NAME=${PACKAGE#$P_PATH/*}
	DES_DIR=${2:-$DEFAULT_DES_DIR}/pkg/$P_PATH

	mkdir -p $DES_DIR

	go build -x -o $DES_DIR/$P_NAME.a $PACKAGE
}

_source_collect() {
	PACKAGE=$1
	P_PATH=${PACKAGE%/*}
	P_NAME=${PACKAGE#$P_PATH/*}
	DES_DIR=${2:-$DEFAULT_DES_DIR}/src/$PACKAGE
	FILE=$DES_DIR/fake.go

	mkdir -p $DES_DIR

	echo "//go:binary-only-package" > $FILE
	echo >> $FILE

	echo "package $P_NAME" >> $FILE
	echo >> $FILE

	echo "import (" >> $FILE
	go list -f '{{ join .Imports "\n" }}' $PACKAGE > $FILE.tmp

	pattern="vendor/"
	while read -r line; do
		rest=${line#*$pattern}	
		echo "	\"$rest\"" >> $FILE
	done < $FILE.tmp

	rm -f $FILE.tmp

	echo ")" >> $FILE

	echo "NOTICE: replace $PACKAGE directory with $DES_DIR"
}

_build() {
	TARGET=$1
	WORK_DIR=$2
	PWD=`pwd`

	if [ ! -d ./cmd/$TARGET ]; then
		echo open ./cmd/$TARGET: directory not exists
		exit 1
	fi

	echo "go build -x -n ./cmd/$TARGET 1>./$TARGET.cmd 2>&1"
	go build -x -n -o $PWD/$TARGET ./cmd/$TARGET 1>./$TARGET.cmd 2>&1

	echo "sed 's#\$WORK#$WORK_DIR#g' ./$TARGET.cmd > ./$TARGET.tmp.cmd"
	sed "s#\$WORK#$WORK_DIR#g" ./$TARGET.cmd > ./$TARGET.tmp.cmd

	echo "bash -x ./$TARGET.tmp.cmd"
	bash -x ./$TARGET.tmp.cmd

	rm $TARGET.cmd
	rm $TARGET.tmp.cmd
}

case "$TYPE" in 
	build)
		_build $*
	;;

	binary-collect)
		_print_start 'Binaray Collect'
		_binary_collect $*
		_print_end
	;;

	src-collect)
		_print_start "Source Collect"
		_source_collect $*
		_print_end
	;;

	collect)
		_print_start 'Binaray Collect'
		_binary_collect $*
		_print_end

		_print_start "Source Collect"
		_source_collect $*
		_print_end
	;;

	*)
		_print_start "Usage"
		echo "$0 [command] [options]"
		echo "COMMAND:"
		echo "	build		Build Target with library"
		echo "	collect		Collect source into fake .go files"
		_print_end
	;;
esac
