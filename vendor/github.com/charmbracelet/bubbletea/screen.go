package tea

import (
	"fmt"
	"io"

	te "github.com/muesli/termenv"
)

func clearLine(w io.Writer) {
	fmt.Fprintf(w, te.CSI+te.EraseLineSeq, 2)
}

func cursorUp(w io.Writer) {
	fmt.Fprintf(w, te.CSI+te.CursorUpSeq, 1)
}

func cursorDown(w io.Writer) {
	fmt.Fprintf(w, te.CSI+te.CursorDownSeq, 1)
}

func insertLine(w io.Writer, numLines int) {
	fmt.Fprintf(w, te.CSI+"%dL", numLines)
}

func moveCursor(w io.Writer, row, col int) {
	fmt.Fprintf(w, te.CSI+te.CursorPositionSeq, row, col)
}

func changeScrollingRegion(w io.Writer, top, bottom int) {
	fmt.Fprintf(w, te.CSI+te.ChangeScrollingRegionSeq, top, bottom)
}

func cursorBack(w io.Writer, n int) {
	fmt.Fprintf(w, te.CSI+te.CursorBackSeq, n)
}
