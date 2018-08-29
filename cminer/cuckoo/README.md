# Readme

The Cuckoo Cycle part is adapted from tromp's repo.

The graph size corresponds to edgebits M and nodebits N = M+1. According to the original makefile, the compiling parameters are given as:


| edgebits | xbits | compression round|
-|-|-
|29|7|14|
|27|6|10|
|24|5|/|
|19|2|/|
|15|0|/|

With a relatationship
edgebits - 2*xbits >= 14
to ensure the valid size of buckets


