#!/usr/bin/env bash

# running the job for 5 hours
let SLEEP_TIME=1*60*60

# GOFLAGS=-modcacherw is required for our CI
# to be able to remove go modules cache
GOFLAGS=-modcacherw make

echo "running cortex..."
#./test.sh > nightly.log 2>&1 &
rm -rf /tmp/.cortex_test
./build/bin/cortex --datadir=/tmp/.cortex_test/ --port=0 --storage.mode=lazy --storage.dht > nightly.log 2>&1 &

CORTEX_PID=$!

echo "sleeping for $SLEEP_TIME seconds $CORTEX_PID"

sleep $SLEEP_TIME

echo "killing CORTEX (pid=$CORTEX_PID)"
kill $CORTEX_PID
echo "boom"

wait $CORTEX_PID

CORTEX_STATUS=$?
echo "The exit status of the process was $CORTEX_STATUS"

exit $CORTEX_STATUS
