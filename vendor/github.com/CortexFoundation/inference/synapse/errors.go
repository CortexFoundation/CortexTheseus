package synapse

import (
	"errors"
)

var (
	KERNEL_RUNTIME_ERROR = errors.New("cvm kernel runtime error")
	KERNEL_LOGIC_ERROR   = errors.New("cvm kernel logic error")
)
