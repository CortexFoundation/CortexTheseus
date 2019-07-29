./cvm.sh > /dev/null 2>&1 &
./node.sh &
while true; do
        server=`ps aux | grep 'cortex cvm' | grep -v grep | grep -v echo`
        if [ ! "$server" ]; then
            ./cvm.sh > /dev/null 2>&1 &
        fi
        sleep 3
done
