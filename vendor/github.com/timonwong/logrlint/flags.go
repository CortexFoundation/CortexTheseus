package logrlint

import (
	"fmt"
	"strings"
)

type loggerCheckersFlag struct {
	stringSet
}

// Set implements flag.Value interface.
func (f *loggerCheckersFlag) Set(s string) error {
	s = strings.TrimSpace(s)
	if s == "" {
		f.stringSet = nil
		return nil
	}

	parts := strings.Split(s, ",")
	set := newStringSet(parts...)
	err := validateIgnoredLoggerFlag(set)
	if err != nil {
		return err
	}
	f.stringSet = set
	return nil
}

// String implements flag.Value interface
func (f *loggerCheckersFlag) String() string {
	return strings.Join(f.List(), ",")
}

func validateIgnoredLoggerFlag(set stringSet) error {
	for key := range set {
		if !loggerCheckersByName.HasKey(key) {
			return fmt.Errorf("unknown logger: %q", key)
		}
	}

	return nil
}
