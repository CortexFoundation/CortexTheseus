#!/bin/bash
set +m
cvm_pid=cvm.pid
node_pid=node.pid
tracker_pid=tracker_pid
remove=1
trap 'EXIT' INT
function EXIT(){
	if [ $remove -eq 0 ];then
		exit 0
	fi
	pkill -P `cat cvm.pid`
	wait $!
	#echo "cvm stopped successfully"
	echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex virtual machine exited with status $?."
	#wait `cat cvm.pid` 2>/dev/null
	#ps aux | grep 'cortex cvm' | grep -v grep | grep -v echo | cut -c 9-15 | xargs kill -9
#	pkill -P `cat tracker.pid`
	pkill -P `cat node.pid`
	wait $!
        #echo "cortex node stopped successfully"
	echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex full node exited with status $?."
	rm -rf cvm.pid node.pid
	wait $!
	echo "INFO [$(date +"%d-%m|%H:%M:%S:000")] Cortex workspace cleaned with status $?."
	#echo "clean [$cvm_pid, $node_pid]"
	#	ps aux | grep 'bittorrent-tracker' | grep -v grep | grep -v echo | cut -c 9-15 | xargs kill -9
	exit 0
}

if [ -f $cvm_pid ];then
	echo "cvm is running with pid [`cat cvm.pid`]. You should stop this process first."
	remove=0
	exit 0
fi

#if [ -f $tracker_pid ];then
#        echo "tracker is running."
#	remove=0
#        exit 0
#fi

if [ -f $node_pid ];then
        echo "cortex node is running with pid [`cat node.pid`]. You should stop this process first."
	remove=0
        exit 0
fi

function start_cvm(){
	#./cvm.sh | grep -v 'Terminated   ' & #>/dev/null 2>&1 &
	#echo $$ > cvm.pid
	./cvm.sh 2>/dev/null &
	echo $! > cvm.pid
}

function start_node(){
	./node.sh &
	echo $! > node.pid
}

#function start_tracker(){
#	./tracker.sh &
#}
#./cvm.sh & 
#echo $! > cvm.pid
start_cvm
#./tracker.sh &
#start_tracker
#./node.sh &
#echo $! > node.pid
start_node
while true; do
        server=`ps aux | grep 'cortex cvm' | grep -v grep | grep -v echo`
        if [ ! "$server" ]; then
            #./cvm.sh &
	    #echo $! > cvm.pid
	    start_cvm
        fi
        sleep 3
done
exit 0 #Exit with success
