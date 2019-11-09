package tagflag

import "fmt"

// The error returned if there are fields in a struct after ExcessArgs.
var ErrFieldsAfterExcessArgs = fmt.Errorf("field(s) after %T", ExcessArgs{})

// This should be added to the end of a struct to soak up any arguments that didn't fit sooner.
type ExcessArgs []string
