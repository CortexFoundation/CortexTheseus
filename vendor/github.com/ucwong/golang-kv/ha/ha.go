package ha

import (
	"fmt"
	"github.com/ucwong/golang-kv/badger"
	"github.com/ucwong/golang-kv/bolt"
	"github.com/ucwong/golang-kv/leveldb"
	"sync"
	"time"
)

type Ha struct {
	ldb *leveldb.LevelDB
	bgr *badger.Badger
	bot *bolt.Bolt
	wg  sync.WaitGroup
}

func Open(path string, level int) *Ha {
	if len(path) == 0 {
		//path = ".ha"
	}

	ha := &Ha{}
	switch level {
	case 0:
		ha.bgr = badger.Open(path)
	case 1:
		ha.bot = bolt.Open(path)
		ha.bgr = badger.Open(path)
	case 2:
		ha.bot = bolt.Open(path)
		ha.bgr = badger.Open(path)
		ha.ldb = leveldb.Open(path)
	default:
		ha.bgr = badger.Open(path)
	}

	if ha.bot == nil && ha.bgr == nil && ha.ldb == nil {
		// suc when one engine is available
		return nil
	}
	fmt.Printf("bolt:%v badger:%v leveldb:%v\n", ha.bot, ha.bgr, ha.ldb)

	return ha
}

func (b *Ha) Get(k []byte) (v []byte) {
	if b.bot != nil {
		v = b.bot.Get(k)
	}
	if v == nil {
		if b.bgr != nil {
			v = b.bgr.Get(k)
		}
	}

	if v == nil {
		if b.ldb != nil {
			v = b.ldb.Get(k)
		}
	}
	return
}

func (b *Ha) Set(k, v []byte) (err error) {
	if b.bot != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.bot.Set(k, v)
		}()
	}
	if b.bgr != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.bgr.Set(k, v)
		}()
	}

	if b.ldb != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.ldb.Set(k, v)
		}()
	}
	b.wg.Wait()
	return
}

func (b *Ha) Del(k []byte) (err error) {
	if b.bot != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.bot.Del(k)
		}()
	}

	if b.bgr != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.bgr.Del(k)
		}()
	}

	if b.ldb != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.ldb.Del(k)
		}()
	}

	b.wg.Wait()

	return
}

func (b *Ha) Prefix(prefix []byte) (res [][]byte) {
	if b.bot != nil {
		res = b.bot.Prefix(prefix)
	}
	if res == nil {
		if b.bgr != nil {
			res = b.bgr.Prefix(prefix)
		}
	}

	if res == nil {
		if b.ldb != nil {
			res = b.ldb.Prefix(prefix)
		}
	}
	return
}

func (b *Ha) Suffix(suffix []byte) (res [][]byte) {
	if b.bot != nil {
		res = b.bot.Suffix(suffix)
	}
	if res == nil {
		if b.bgr != nil {
			res = b.bgr.Suffix(suffix)
		}
	}
	if res == nil {
		if b.ldb != nil {
			res = b.ldb.Suffix(suffix)
		}
	}
	return
}

func (b *Ha) Scan() (res [][]byte) {
	if b.bot != nil {
		res = b.bot.Scan()
	}
	if res == nil {
		if b.bgr != nil {
			res = b.bgr.Scan()
		}
	}
	if res == nil {
		if b.ldb != nil {
			res = b.ldb.Scan()
		}
	}
	return
}

func (b *Ha) SetTTL(k, v []byte, expire time.Duration) (err error) {
	if b.bot != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.bot.SetTTL(k, v, expire)
		}()
	}
	if b.bgr != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.bgr.SetTTL(k, v, expire)
		}()
	}
	if b.ldb != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.ldb.SetTTL(k, v, expire)
		}()
	}

	b.wg.Wait()

	return
}

func (b *Ha) Range(start, limit []byte) (res [][]byte) {
	if b.bot != nil {
		res = b.bot.Range(start, limit)
	}
	if res == nil {
		if b.bgr != nil {
			res = b.bgr.Range(start, limit)
		}
	}

	if res == nil {
		if b.ldb != nil {
			res = b.ldb.Range(start, limit)
		}
	}

	return
}

func (b *Ha) Close() (err error) {
	if b.bot != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.bot.Close()
		}()
	}

	if b.ldb != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.ldb.Close()
		}()
	}

	if b.bgr != nil {
		b.wg.Add(1)
		go func() {
			defer b.wg.Done()
			b.bgr.Close()
		}()
	}

	b.wg.Wait()

	return
}
