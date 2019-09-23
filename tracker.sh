#!/bin/bash
tracker_pid=tracker.pid
if [ -f ${tracker_pid} ];then
        echo "tracker is running."
        exit 0
fi
./build/bin/tracker > /dev/null 2>&1 &
#echo $! > ${tracker_pid}
echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex tracker start with status $?."
echo $! > ${tracker_pid}
chmod 644 "${tracker_pid}"
