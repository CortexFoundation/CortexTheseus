# filecache
## A simple file cache of pure Golang

## Install

First, you need to install the package:

```
go get -u github.com/ucwong/filecache
```

## Overview

A file cache can be created with either the `NewDefaultCache()` function to
get a cache with the defaults set, or `NewCache()` to get a new cache with
`0` values for everything; you will not be able to store items in this cache
until the values are changed; specifically, at a minimum, you should set
the `MaxItems` field to be > 0.

Let's start with a basic example; we'll create a basic cache and give it a
maximum item size of 128M:

```
import (
      "github.com/ucwong/filecache"
)

...

cache := filecache.NewDefaultCache()
cache.MaxSize = 128 * filecache.Megabyte
cache.Start()

...

cache.Stop()

```

The `Kilobyte`, `Megabyte`, and `Gigabyte` constants are provided as a
convience when setting cache sizes.

When `cache.Start()` is called, a goroutine is launched in the background
that routinely checks the cache for expired items. The delay between
runs is specified as the number of seconds given by `cache.Every` ("every
`cache.Every` seconds, check for expired items"). There are three criteria
used to determine whether an item in the cache should be expired; they are:

   1. Has the file been modified on disk? (The cache stores the last time
      of modification at the time of caching, and compares that to the
      file's current last modification time).
   2. Has the file been in the cache for longer than the maximum allowed
      time?
   3. Is the cache at capacity? When a file is being cached, a check is
      made to see if the cache is currently filled. If it is, the item that
      was last accessed the longest ago is expired and the new item takes
      its place. When loading items asynchronously, this check might miss
      the fact that the cache will be at capacity; the background scanner
      performs a check after its regular checks to ensure that the cache is
      not at capacity.

The background scanner can be disabled by setting `cache.Every` to 0; if so,
cache expiration is only done when the cache is at capacity.

Once the cache is no longer needed, a call to `cache.Stop()` will close down
the channels and signal the background scanner that it should stop.
