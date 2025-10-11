#!/bin/bash
set -euo pipefail

log() {
  echo "[`date '+%Y-%m-%d %H:%M:%S'`] $*"
}

error_handler() {
  local exit_code=$?
  log "[ERROR] Script failed at line $1 with exit code $exit_code"
  exit $exit_code
}

trap 'error_handler $LINENO' ERR

for cmd in git make tar zip md5sum sha256sum; do
  if ! command -v $cmd >/dev/null 2>&1; then
    log "[ERROR] Command '$cmd' not found. Please install it first."
    exit 1
  fi
done

cd ..

log "Fetching latest tags from origin"
git fetch --tags origin

version=$(git tag --sort=committerdate | tail -1)
if [ -z "$version" ]; then
  log "[ERROR] No git tag found!"
  exit 1
fi

log "Checking out git tag: $version"
git checkout "$version"

commit=$(git rev-parse HEAD | cut -c 1-8)
prefix="cortex-$(uname -s)-$(uname -p)"
name="${prefix}-${version}-${commit}"

log "Building release: $name"
make clean
make -j"$(nproc)" >/dev/null 2>&1
./build/bin/cortex version

if ! command -v zip >/dev/null 2>&1; then
  log "Installing zip package"
  DEBIAN_FRONTEND=noninteractive sudo apt-get update -qq >/dev/null 2>&1
  DEBIAN_FRONTEND=noninteractive sudo apt-get install -y zip >/dev/null 2>&1
fi

cd release
log "Cleaning old release files"
rm -rf "${prefix}"* *.tar.gz *.zip checksum

log "Preparing release directory"
mkdir -p "${name}/plugins"
cp ../build/bin/cortex "${name}"
cp ../plugins/* "${name}/plugins/"

log "Creating ${name}.tar.gz"
tar zcf "${name}.tar.gz" "${name}"

log "Creating ${name}.zip"
zip -rq "${name}.zip" "${name}"

log "Generating checksum file"
{
  echo "MD5"
  md5sum "${name}.tar.gz" "${name}.zip"
  echo "SHA256"
  sha256sum "${name}.tar.gz" "${name}.zip"
} > checksum

cat checksum
log "Release completed successfully!"
