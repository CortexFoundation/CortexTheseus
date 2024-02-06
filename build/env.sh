#!/bin/sh

set -e

if [ ! -f "build/env.sh" ]; then
    echo "$0 must be run from the root of the repository."
    exit 2
fi

# Create fake Go workspace if it doesn't exist yet.
workspace="$PWD/build/_workspace"
root="$PWD"
dir="$workspace/src/github.com/CortexFoundation"
if [ ! -L "$dir/CortexTheseus" ]; then
    mkdir -p "$dir"
    cd "$dir"
    ln -s ../../../../../. CortexTheseus
    cd "$root"
fi

# Set up the environment to use the workspace.
GOPATH="$workspace"
export GOPATH
export GO111MODULE=auto

# Run the command inside the workspace.
cd "$dir/CortexTheseus"
PWD="$dir/CortexTheseus"

git submodule init
git submodule update

# Launch the arguments with the configured environment.
exec "$@"
