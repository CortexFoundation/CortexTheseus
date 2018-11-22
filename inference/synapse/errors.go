package synapse

import (
	"errors"
)

var (
	ErrBuiltInTorrentFsPrefix = "BUILT-IN TORRENT FS ERROR"

	ErrInputFileNotExistFlag = "[ " + ErrBuiltInTorrentFsPrefix + " ] input file not exist"
	ErrModelFileNotExistFlag = "[ " + ErrBuiltInTorrentFsPrefix + " ] model file not exist"

	ErrInputFileNotExist = errors.New(ErrInputFileNotExistFlag)
	ErrModelFileNotExist = errors.New(ErrModelFileNotExistFlag)

	ErrFatal = errors.New("fatal")
)

// If infered by local, check struct error is enough.
// Else infered by remote, must be checked by error's text
func CheckBuiltInTorrentFsError(err error) bool {
	if err == nil {
		return false
	}

	errStr := err.Error()
	return errStr == ErrInputFileNotExistFlag ||
		errStr == ErrModelFileNotExistFlag || errStr == ErrFatal.Error()
}
