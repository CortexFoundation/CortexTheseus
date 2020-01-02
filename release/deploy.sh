name=$1
rm -rf ${name}
rm -rf ${name}.tar.gz
rm -rf ${name}.zip
#mkdir ${name}
mkdir -p ${name}/plugins
cp ../build/bin/cortex ${name}
cp ../plugins/* ${name}/plugins
tar zvcf ${name}.tar.gz ${name}
zip -r ${name}.zip ${name}
md5sum ${name}.tar.gz
md5sum ${name}.zip
