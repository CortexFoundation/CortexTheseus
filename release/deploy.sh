#!/bin/sh

cd ..
git fetch origin
git tag --sort=committerdate | tail -1

while read -p "... ... Please input the release version number :" version
do
if  [ ! -n "$version" ] ;then
    echo "You have not input a release version number!"
else
    break
fi
done

echo "... ... Checkout git tag $version"
git checkout $version

while read -p "... ... Please input the latest commit prefix :" commit
do
if  [ ! -n "$commit" ] ;then
    echo "You have not input a commit prefix!"
else
    break
fi
done

prefix=cortex-linux-amd64
name=${prefix}-${version}-${commit}

echo "... ... Building release ${name}"
make clean && make -j$(nproc) > /dev/null 2>&1
./build/bin/cortex version

apt install zip > /dev/null 2>&1
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
echo "MD5" >> checksum
md5sum ${name}.tar.gz >> checksum
md5sum ${name}.zip >> checksum
echo "SHA256" >> checksum
sha256sum ${name}.tar.gz >> checksum
sha256sum ${name}.zip >> checksum
cat checksum
