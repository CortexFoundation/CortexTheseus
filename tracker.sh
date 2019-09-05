#!/bin/sh
./build/bin/tracker &
echo $! > tracker.pid
