./cvm.sh > /dev/null 2>&1 &
./node.sh &
while true; do
        server=`ps aux | grep 'cortex cvm' | grep -v grep`
        if [ ! "$server" ]; then
            ./cvm.sh > /dev/null 2>&1 &
            sleep 10
        fi
        sleep 5
done
