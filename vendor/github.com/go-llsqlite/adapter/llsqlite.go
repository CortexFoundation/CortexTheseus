package sqlite

import (
	g "github.com/anacrolix/generics"
)

type ColIter int

func (me *ColIter) PostInc() int {
	ret := int(*me)
	*me++
	return ret
}

func (me ColIter) Get() int {
	return int(me)
}

func IsResultCode(err error, code ResultCode) bool {
	actual, ok := GetResultCode(err)
	return ok && actual == code
}

func IsPrimaryResultCodeErr(err error, code ResultCode) bool {
	actual, ok := GetResultCode(err)
	return ok && actual.ToPrimary() == code.ToPrimary()
}

func StmtColumnOption[V any](stmt *Stmt, col int, get func(_ *Stmt, col int) V, opt *g.Option[V]) {
	if stmt.ColumnType(col) == TypeNull {
		opt.SetNone()
		return
	}
	opt.Set(get(stmt, col))
}
