package accounts

import "testing"

func TestNewAuthNeededError(t *testing.T) {
	neededString := "this error"
	neededErrorString := "authentication needed: this error"
	if err := NewAuthNeededError(neededString); err.Error() != neededErrorString {
		t.Errorf("the error string should be: %v, got %v", neededErrorString, err.Error())
	}
}
