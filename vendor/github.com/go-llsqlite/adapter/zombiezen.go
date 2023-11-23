//go:build zombiezen_sqlite

package sqlite

import "zombiezen.com/go/sqlite"

type (
	Conn         = sqlite.Conn
	Stmt         = sqlite.Stmt
	FunctionImpl = sqlite.FunctionImpl
	Context      = sqlite.Context
	Value        = sqlite.Value
	ResultCode   sqlite.ResultCode
	Blob         = sqlite.Blob
)

const (
	TypeNull = sqlite.TypeNull

	OpenNoMutex     = sqlite.OpenNoMutex
	OpenReadOnly    = sqlite.OpenReadOnly
	OpenURI         = sqlite.OpenURI
	OpenWAL         = sqlite.OpenWAL
	OpenCreate      = sqlite.OpenCreate
	OpenReadWrite   = sqlite.OpenReadWrite
	OpenSharedCache = sqlite.OpenSharedCache
	// I don't see this in the version of zombiezen I'm currently using. It might be in an updated
	// version. Here's the documentation about it from
	// https://sqlite.org/capi3ref.html#sqlite3_open:
	//
	// Note in particular that the SQLITE_OPEN_EXCLUSIVE flag is a no-op for sqlite3_open_v2(). The
	// SQLITE_OPEN_EXCLUSIVE does *not* cause the open to fail if the database already exists. The
	// SQLITE_OPEN_EXCLUSIVE flag is intended for use by the VFS interface only, and not by
	// sqlite3_open_v2().
	//
	// This would suggest it's okay to set it to 0 if there's no appropriate value for it.
	//OpenExclusive   = 0

	ResultCodeConstraintUnique = sqlite.ResultConstraintUnique
	ResultCodeInterrupt        = sqlite.ResultInterrupt
	ResultCodeOk               = sqlite.ResultOK
	ResultCodeAbort            = sqlite.ResultAbort
	ResultCodeGenericError     = sqlite.ResultError
)

var (
	BlobValue = sqlite.BlobValue
	OpenConn  = sqlite.OpenConn
	ErrCode   = sqlite.ErrCode
)

// This produces an error code even if it's not an underlying sqlite error. This could differ from
// the crawshaw implementation.
func GetResultCode(err error) (ResultCode, bool) {
	return sqlite.ErrCode(err), true
}
