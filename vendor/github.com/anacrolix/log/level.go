package log

type Level int

const (
	Debug Level = iota - 1
	Info
	Warning
	Error
	Critical
	Fatal
)
