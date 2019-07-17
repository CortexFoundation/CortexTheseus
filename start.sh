./cvm.sh &
./node.sh &
while true; do
        server=`ps aux | grep cvm | grep -v grep`
        if [ ! "$server" ]; then
            ./cvm.sh &
            sleep 10
        fi
        sleep 5
done
