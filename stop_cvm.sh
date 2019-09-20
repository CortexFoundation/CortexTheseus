#!/bin/bash
cvm_pid=cvm.pid
function EXIT(){

	if [ -f ${cvm_pid} ];then
		pkill -P `cat ${cvm_pid}`
		wait $!
		echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex virtual machine exited with status $?."
	else
		echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex virtual machine pid file not exist. Try to stop process manually."
	fi
	rm -rf ${cvm_pid}
	wait $!
	echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex workspace cleaned with status $?."
	exit 0
}

EXIT

exit 0 #Exit with success
