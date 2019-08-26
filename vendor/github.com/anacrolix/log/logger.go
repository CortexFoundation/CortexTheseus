package log

import "fmt"

type Logger struct {
	hs      map[Handler]struct{}
	values  map[interface{}]struct{}
	filters map[*Filter]struct{}
}

func (l *Logger) SetHandler(h Handler) {
	l.hs = map[Handler]struct{}{h: struct{}{}}
}

func (l *Logger) Clone() *Logger {
	ret := &Logger{
		hs:      make(map[Handler]struct{}),
		values:  make(map[interface{}]struct{}),
		filters: make(map[*Filter]struct{}),
	}
	for h, v := range l.hs {
		ret.hs[h] = v
	}
	for v, v_ := range l.values {
		ret.values[v] = v_
	}
	for f := range l.filters {
		ret.filters[f] = struct{}{}
	}
	return ret
}

func (l *Logger) AddValue(v interface{}) *Logger {
	l.values[v] = struct{}{}
	return l
}

// rename Log to allow other implementers
func (l *Logger) Handle(m Msg) {
	for v := range l.values {
		m.AddValue(v)
	}
	for f := range l.filters {
		if !f.ff(&m) {
			return
		}
	}
	for h := range l.hs {
		h.Emit(m)
	}
}

func (l *Logger) AddFilter(f *Filter) *Logger {
	if l.filters == nil {
		l.filters = make(map[*Filter]struct{})
	}
	l.filters[f] = struct{}{}
	return l
}

// Helper for compatibility with "log".Logger.
func (l *Logger) Printf(format string, a ...interface{}) {
	l.Handle(Fmsg(format, a...).Skip(1))
}

// Helper for compatibility with "log".Logger.
func (l *Logger) Print(v ...interface{}) {
	l.Handle(Str(fmt.Sprint(v...)).Skip(1))
}
