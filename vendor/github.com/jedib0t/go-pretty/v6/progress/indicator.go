package progress

import (
	"strings"
	"sync/atomic"
	"time"

	"github.com/jedib0t/go-pretty/v6/text"
)

// IndeterminateIndicator defines the structure for the indicator to indicate
// indeterminate progress. Ex.: {0, <=>}
type IndeterminateIndicator struct {
	Position int
	Text     string
}

// IndeterminateIndicatorGenerator is a function that takes the maximum length
// of the progress bar and returns an IndeterminateIndicator telling the
// indicator string, and the location of the same in the progress bar.
//
// Technically, this could generate and return the entire progress bar string to
// override the full display of the same - this is done by the Dominoes and
// Pac-Man examples below.
type IndeterminateIndicatorGenerator func(maxLen int) IndeterminateIndicator

// IndeterminateIndicatorDominoes simulates a bunch of dominoes falling back and
// forth.
func IndeterminateIndicatorDominoes(duration time.Duration) IndeterminateIndicatorGenerator {
	return timedIndeterminateIndicatorGenerator(indeterminateIndicatorDominoes(), duration)
}

// IndeterminateIndicatorMovingBackAndForth incrementally moves from the left to
// right and back for each specified duration. If duration is 0, then every
// single invocation moves the indicator.
func IndeterminateIndicatorMovingBackAndForth(indicator string, duration time.Duration) IndeterminateIndicatorGenerator {
	return timedIndeterminateIndicatorGenerator(indeterminateIndicatorMovingBackAndForth(indicator), duration)
}

// IndeterminateIndicatorMovingLeftToRight incrementally moves from the left to
// right and starts from left again for each specified duration. If duration is
// 0, then every single invocation moves the indicator.
func IndeterminateIndicatorMovingLeftToRight(indicator string, duration time.Duration) IndeterminateIndicatorGenerator {
	return timedIndeterminateIndicatorGenerator(indeterminateIndicatorMovingLeftToRight(indicator), duration)
}

// IndeterminateIndicatorMovingRightToLeft incrementally moves from the right to
// left and starts from right again for each specified duration. If duration is
// 0, then every single invocation moves the indicator.
func IndeterminateIndicatorMovingRightToLeft(indicator string, duration time.Duration) IndeterminateIndicatorGenerator {
	return timedIndeterminateIndicatorGenerator(indeterminateIndicatorMovingRightToLeft(indicator), duration)
}

// IndeterminateIndicatorPacMan simulates a Pac-Man character chomping through
// the progress bar back and forth.
func IndeterminateIndicatorPacMan(duration time.Duration) IndeterminateIndicatorGenerator {
	return timedIndeterminateIndicatorGenerator(indeterminateIndicatorPacMan(), duration)
}

// IndeterminateIndicatorColoredDominoes simulates a bunch of colored dominoes falling back and
// forth.
func IndeterminateIndicatorColoredDominoes(duration time.Duration, slashColor, backslashColor text.Color) IndeterminateIndicatorGenerator {
	baseGen := IndeterminateIndicatorDominoes(duration)
	return func(maxLen int) IndeterminateIndicator {
		base := baseGen(maxLen)
		colored := strings.Builder{}
		for _, ch := range base.Text {
			switch ch {
			case '/':
				colored.WriteString(text.Colors{slashColor}.Sprint(string(ch)))
			case '\\':
				colored.WriteString(text.Colors{backslashColor}.Sprint(string(ch)))
			default:
				colored.WriteRune(ch)
			}
		}
		return IndeterminateIndicator{
			Position: 0,
			Text:     colored.String(),
		}
	}
}

// IndeterminateIndicatorPacManChomp simulates a Pac-Man character chomping through the progress
// bar back and forth.
func IndeterminateIndicatorPacManChomp(duration time.Duration) IndeterminateIndicatorGenerator {
	return timedIndeterminateIndicatorGenerator(indeterminateIndicatorPacManChomp(), duration)
}

func indeterminateIndicatorPacManChomp() IndeterminateIndicatorGenerator {
	var frame int64

	return func(maxLen int) IndeterminateIndicator {
		i := atomic.AddInt64(&frame, 1)
		cycle := i / int64(maxLen-1)
		pos := int(i % int64(maxLen-1))

		leftToRight := cycle%2 == 0
		if !leftToRight {
			pos = (maxLen - 1) - pos
		}

		// Alternate between open and closed mouth
		mouthOpen := (i/3)%2 == 0
		pac := "c"
		if !leftToRight {
			pac = "ɔ"
		}
		if !mouthOpen {
			pac = "●"
		}

		trail := make([]string, maxLen)
		for j := 0; j < maxLen; j++ {
			trail[j] = "·"
		}

		for j := 0; j < maxLen; j++ {
			if (leftToRight && j < pos) || (!leftToRight && j > pos) {
				trail[j] = " "
			}
		}

		trail[pos] = pac

		var line string
		for j := 0; j < maxLen; j++ {
			switch {
			case j == pos:
				line += text.Colors{text.FgHiYellow}.Sprint(trail[j])
			case trail[j] == "·":
				line += text.Colors{text.FgWhite}.Sprint(trail[j])
			default:
				line += trail[j]
			}
		}

		return IndeterminateIndicator{
			Position: 0,
			Text:     line,
		}
	}
}

func indeterminateIndicatorDominoes() IndeterminateIndicatorGenerator {
	direction := 1 // positive == left to right; negative == right to left
	nextPosition := 0

	out := strings.Builder{}
	generateIndicator := func(currentPosition int, maxLen int) string {
		out.Reset()
		out.WriteString(strings.Repeat("/", currentPosition))
		out.WriteString(strings.Repeat("\\", maxLen-currentPosition))
		return out.String()
	}

	return func(maxLen int) IndeterminateIndicator {
		currentPosition := nextPosition

		if currentPosition == 0 {
			direction = 1
		} else if currentPosition == maxLen {
			direction = -1
		}
		nextPosition += direction

		return IndeterminateIndicator{
			Position: 0,
			Text:     generateIndicator(currentPosition, maxLen),
		}
	}
}

func indeterminateIndicatorMovingBackAndForth(indicator string) IndeterminateIndicatorGenerator {
	direction := 1 // positive == left to right; negative == right to left
	nextPosition := 0

	return func(maxLen int) IndeterminateIndicator {
		currentPosition := nextPosition

		if currentPosition == 0 {
			direction = 1
		} else if currentPosition+text.StringWidthWithoutEscSequences(indicator) == maxLen {
			direction = -1
		}
		nextPosition += direction

		return IndeterminateIndicator{
			Position: currentPosition,
			Text:     indicator,
		}
	}
}

func indeterminateIndicatorMovingLeftToRight(indicator string) IndeterminateIndicatorGenerator {
	nextPosition := 0

	return func(maxLen int) IndeterminateIndicator {
		currentPosition := nextPosition

		nextPosition++
		if nextPosition+text.StringWidthWithoutEscSequences(indicator) > maxLen {
			nextPosition = 0
		}

		return IndeterminateIndicator{
			Position: currentPosition,
			Text:     indicator,
		}
	}
}

func indeterminateIndicatorMovingRightToLeft(indicator string) IndeterminateIndicatorGenerator {
	nextPosition := -1

	return func(maxLen int) IndeterminateIndicator {
		if nextPosition == -1 {
			nextPosition = maxLen - text.StringWidthWithoutEscSequences(indicator)
		}
		currentPosition := nextPosition
		nextPosition--

		return IndeterminateIndicator{
			Position: currentPosition,
			Text:     indicator,
		}
	}
}

func indeterminateIndicatorPacMan() IndeterminateIndicatorGenerator {
	pacManMovingRight, pacManMovingLeft := "ᗧ", "ᗤ"
	direction := 1 // positive == left to right; negative == right to left
	indicator := pacManMovingRight
	nextPosition := 0

	out := strings.Builder{}
	generateIndicator := func(currentPosition int, maxLen int) string {
		out.Reset()
		if currentPosition > 0 {
			out.WriteString(strings.Repeat(" ", currentPosition))
		}
		out.WriteString(indicator)
		out.WriteString(strings.Repeat(" ", maxLen-currentPosition-1))
		return out.String()
	}

	return func(maxLen int) IndeterminateIndicator {
		currentPosition := nextPosition
		currentText := generateIndicator(currentPosition, maxLen)

		if currentPosition == 0 {
			direction = 1
			indicator = pacManMovingRight
		} else if currentPosition+text.StringWidthWithoutEscSequences(indicator) == maxLen {
			direction = -1
			indicator = pacManMovingLeft
		}
		nextPosition += direction

		return IndeterminateIndicator{
			Position: 0,
			Text:     currentText,
		}
	}
}

// timedIndeterminateIndicatorGenerator ticks based on the given duration. If
// duration is 0, it ticks for every invocation.
func timedIndeterminateIndicatorGenerator(indicatorGenerator IndeterminateIndicatorGenerator, duration time.Duration) IndeterminateIndicatorGenerator {
	var indeterminateIndicator *IndeterminateIndicator
	lastRenderTime := time.Now()

	return func(maxLen int) IndeterminateIndicator {
		currRenderTime := time.Now()
		if indeterminateIndicator == nil || duration == 0 || currRenderTime.Sub(lastRenderTime) > duration {
			tmpIndeterminateIndicator := indicatorGenerator(maxLen)
			indeterminateIndicator = &tmpIndeterminateIndicator
			lastRenderTime = currRenderTime
		}

		return *indeterminateIndicator
	}
}
