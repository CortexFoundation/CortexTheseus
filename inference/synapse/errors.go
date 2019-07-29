package synapse

import (
	"errors"
)

var (
	KERNEL_RUNTIME_ERROR = errors.New("Kernel runtime error")
	KERNEL_LOGIC_ERROR   = errors.New("Kernel logic error")
)
