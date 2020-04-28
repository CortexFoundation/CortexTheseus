package torrentfs

import (
	"strconv"
)

func ProgressBar(x, y int64, desc string) string {
	progress := ""
	for i := 10; i > 0; i-- {
		if int64(i) > (10*x)/y {
			progress = progress + " "
		} else {
			progress = progress + "<"
		}
	}

	prog := float64(x*100) / float64(y)
	f := strconv.FormatFloat(prog, 'f', 2, 64)
	return "[ " + progress + " ] " + f + "% " + desc
}
