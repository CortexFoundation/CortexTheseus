#!/bin/bash
trap 'EXIT' INT
function EXIT(){
	pkill -P `cat cvm.pid`
	ps aux | grep 'cortex cvm' | grep -v grep | grep -v echo | cut -c 9-15 | xargs kill -9
	pkill -P `cat tracker.pid`
	#	ps aux | grep 'bittorrent-tracker' | grep -v grep | grep -v echo | cut -c 9-15 | xargs kill -9
	exit 0
}

./cvm.sh & #> /dev/null 2>&1 &
echo $! > cvm.pid
./node.sh &
./tracker.sh &
while true; do
        server=`ps aux | grep 'cortex cvm' | grep -v grep | grep -v echo`
        if [ ! "$server" ]; then
            ./cvm.sh & #> /dev/null 2>&1 &
	    echo $! > cvm.pid
        fi
        sleep 3
done
exit 0 #Exit with success
