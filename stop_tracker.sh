#!/bin/bash
tracker_pid=tracker.pid
if [ ! -f ${tracker_pid} ];then
        echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex tracker is not running."
        exit 0
fi
pkill -P `cat ${tracker_pid}`
wait $!
echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex tracker exited with status $?."
rm -rf ${tracker_pid}
echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex workspace cleaned with status $?."
