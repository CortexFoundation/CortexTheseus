#!/bin/bash
main_pid=main.pid

if [ -f ${main_pid} ];then
                kill -s 9 `cat ${main_pid}`
                wait $!
                echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex main process exited with status $?."
		rm -rf ${main_pid}
        else
                echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex main process pid file not exist. Try to stop process manually."
		exit 0
fi
./stop_node.sh
./stop_cvm.sh
