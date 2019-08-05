./cvm_gpu.sh > /dev/null 2>&1 &
./solo.sh &
while true; do
        server=`ps aux | grep 'cortex cvm' | grep -v grep | grep -v echo`
        if [ ! "$server" ]; then
            ./cvm_gpu.sh > /dev/null 2>&1 &
        fi
        sleep 3
done
