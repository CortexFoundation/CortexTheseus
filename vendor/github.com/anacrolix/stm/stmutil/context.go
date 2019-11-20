package stmutil

import (
	"context"

	"github.com/anacrolix/stm"
)

func ContextDoneVar(ctx context.Context) (*stm.Var, func()) {
	if ctx.Err() != nil {
		return stm.NewVar(true), func() {}
	}
	ctx, cancel := context.WithCancel(ctx)
	_var := stm.NewVar(false)
	go func() {
		<-ctx.Done()
		stm.AtomicSet(_var, true)
	}()
	return _var, cancel
}
