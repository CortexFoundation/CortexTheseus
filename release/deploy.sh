#!/bin/sh

read -p "Please input the release version :" version
echo "... ... Building release"
cd ..
git checkout version
make clean && make -j8
./build/bin/cortex version
read -p "Please input the latest commit :" commit
apt install zip
#$version=$1
#commit=$2
prefix=cortex-linux-amd64
name=${prefix}-${version}-${commit}
echo "... ... Release space clean up"
cd release
rm -rf ${prefix}*
rm -rf *.tar.gz
rm -rf *.zip
rm -rf checksum

echo "... ... Release space initialized"
mkdir -p ${name}/plugins
cp ../build/bin/cortex ${name}
cp ../plugins/* ${name}/plugins

echo "... ... Release (${name}.tar.gz) package"
tar zvcf ${name}.tar.gz ${name}
echo "... ... Release (${name}.zip) package"
zip -vr ${name}.zip ${name}

echo "... ... Check sum"
md5sum ${name}.tar.gz >> checksum
md5sum ${name}.zip >> checksum
cat checksum
