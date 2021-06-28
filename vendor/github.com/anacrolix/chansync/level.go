package chansync

import (
	"sync"
)

type LevelTrigger struct {
	ch       chan struct{}
	initOnce sync.Once
}

func (me *LevelTrigger) Signal() chan<- struct{} {
	me.init()
	return me.ch
}

func (me *LevelTrigger) Active() Active {
	me.init()
	return me.ch
}

func (me *LevelTrigger) init() {
	me.initOnce.Do(func() {
		me.ch = make(chan struct{})
	})
}
