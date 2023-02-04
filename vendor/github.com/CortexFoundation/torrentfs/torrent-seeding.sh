#!/bin/bash

./build/bin/torrent-create workspace/data -p=4096 > workspace/test-torrent
./build/bin/torrent-magnet < workspace/test-torrent
mkdir -p mnt/$1/data
cp workspace/test-torrent mnt/$1/torrent
cp -r workspace/data/* mnt/${1}/data
./build/bin/seeding -dataDir=mnt
