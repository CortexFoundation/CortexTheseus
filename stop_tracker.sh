#!/bin/bash
pkill -P `cat tracker.pid`
rm -rf tracker.pid
