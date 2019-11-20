#!/bin/bash
node_pid=node.pid
function EXIT(){

	if [ -f ${node_pid} ];then
		pkill -P `cat ${node_pid}`
		wait $!
		echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex full node exited with status $?."
	else
		echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex full node pid file not exist. Try to stop process manually."
	fi
	rm -rf ${node_pid}
	wait $!
	echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex workspace cleaned with status $?."
	exit 0
}

EXIT

exit 0 #Exit with success
