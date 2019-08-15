#!/bin/sh
./build/bin/cortex --infer.devicetype=remote://localhost:4321 --miner.coinbase=0x553540b55c92ef0db6ff94fe764816acf0c4d9e7 --mine --miner.devices=0 --miner.threads=1 --miner.cuda
