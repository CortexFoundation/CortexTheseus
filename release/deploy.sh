#!/bin/sh

read -p "Please input the release version :" version
echo "... ... Checkout git tag $version" 
cd ..
git fetch origin
git checkout $version

echo "... ... Building release"
make clean && make -j$(nproc) > /dev/null 2>&1
./build/bin/cortex version

read -p "Please input the latest commit :" commit
apt install zip
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
sha256sum ${name}.tar.gz >> checksum
sha256sum ${name}.zip >> checksum
cat checksum
