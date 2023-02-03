#!/bin/bash

./build/bin/torrent-create workspace/data -p=4096 > workspace/test-torrent
./build/bin/torrent-magnet < workspace/test-torrent
