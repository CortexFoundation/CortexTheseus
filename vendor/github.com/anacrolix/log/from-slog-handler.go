package log

import (
	"context"
	"fmt"
	"log/slog"
	"time"
)

// This doesn't seem like a good idea. Handlers in analog occur after filtering, so attrs in the
// slog.Handler won't be used by the analog.Logger.
type SlogHandlerAsHandler struct {
	SlogHandler slog.Handler
}

func (me SlogHandlerAsHandler) Handle(r Record) {
	slogLevel, ok := toSlogLevel(r.Level)
	if !ok {
		panic(r.Level)
	}
	var pc [1]uintptr
	r.Callers(1, pc[:])
	slogRecord := slog.NewRecord(time.Now(), slogLevel, r.Msg.String(), pc[0])
	anyNames := make([]any, 0, len(r.Names))
	for _, name := range r.Names {
		anyNames = append(anyNames, name)
	}
	slogRecord.AddAttrs(slog.Any("names", r.Names))
	r.Values(func(value interface{}) (more bool) {
		if item, ok := value.(item); ok {
			slogRecord.AddAttrs(slog.Any(fmt.Sprint(item.key), item.value))
			return true
		}
		return true
	})
	err := me.SlogHandler.Handle(context.Background(), slogRecord)
	if err != nil {
		panic(err)
	}
}

var _ Handler = SlogHandlerAsHandler{}
