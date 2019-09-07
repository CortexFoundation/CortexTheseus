#!/bin/bash
tracker_pid=tracker.pid
if [ -f ${tracker_pid} ];then
        echo "tracker is running."
        exit 0
fi
./build/bin/tracker > /dev/null 2>&1 &
echo $! > ${tracker_pid}
