#!/bin/bash

TYPE=$1
shift

goversion=`go version`
IFS='  ' read -ra array <<< $goversion
pkgSystemName="${array[@]: -1:1}"
pkgSystemName=`echo "$pkgSystemName" | tr / _`

DEFAULT_DES_DIR=./root/
DEFAULT_BACKUP_DIR=./tmp/

_binary_collect() {
	PACKAGE=$1
	P_PATH=${PACKAGE%/*}
	P_NAME=${PACKAGE#$P_PATH/*}
	DES_DIR=${2:-$DEFAULT_DES_DIR}/${pkgSystemName}/$P_PATH

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

	echo "package $P_NAME" >> $FILE
	echo >> $FILE

	echo "import (" >> $FILE
	go list -f '{{ join .Imports "\n" }}' $PACKAGE > $FILE.tmp

	pattern="vendor/"
	while read -r line; do
		rest=${line#*$pattern}	
		echo "	_ \"$rest\"" >> $FILE
	done < $FILE.tmp

	rm -f $FILE.tmp

	echo ")" >> $FILE

	echo "NOTICE: replace $PACKAGE directory with $DES_DIR"
}

_replace_bin() {
	PACKAGE=$1
	P_PATH=${PACKAGE%/*}
	P_NAME=${PACKAGE#$P_PATH/*}
	DES_DIR=${2:-$DEFAULT_DES_DIR}

	mkdir -p ${GOPATH}/pkg/${pkgSystemName}/${P_PATH}
	cp ${DES_DIR}/${pkgSystemName}/${P_PATH}/${P_NAME}.a ${GOPATH}/pkg/${pkgSystemName}/${P_PATH}
}

_replace_src() {
	PACKAGE=$1
	P_PATH=${PACKAGE%/*}
	P_NAME=${PACKAGE#$P_PATH/*}
	DES_DIR=${2:-$DEFAULT_DES_DIR}

	# make original source backup
	mkdir -p ${DEFAULT_BACKUP_DIR}
	mv -n ${GOPATH}/src/${PACKAGE} ${DEFAULT_BACKUP_DIR}/
	rm -rf ${GOPATH}/src/${PACKAGE}

	cp -r ${DES_DIR}/src/${PACKAGE} ${GOPATH}/src/${PACKAGE}
}

_restore_src() {
	PACKAGE=$1
	P_PATH=${PACKAGE%/*}
	P_NAME=${PACKAGE#$P_PATH/*}
	
	cp -r ${DEFAULT_BACKUP_DIR}/${P_NAME} ${GOPATH}/src/${P_PATH}/
}

case "$TYPE" in 
	binary-collect)
		_binary_collect $*
	;;

	src-collect)
		_source_collect $*
	;;

	collect)
		_source_collect $*
		_binary_collect $*
	;;

	replace_bin)
		_replace_bin $*
	;;

	replace_src)
		_replace_src $*
	;;

	restore_src)
		_restore_src $*
	;;

	*)
		echo "$0 [command] [options]"
		echo "COMMAND:"
		echo "	build		Build Target with library"
		echo "	collect		Collect source into fake .go files"
	;;
esac
