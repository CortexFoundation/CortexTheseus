package tagflag

import "fmt"

var ErrFieldsAfterExcessArgs = fmt.Errorf("field(s) after %T", ExcessArgs{})

type ExcessArgs []string
