package memorydb

import (
	"github.com/CortexFoundation/CortexTheseus/ctxcdb"
	"github.com/CortexFoundation/CortexTheseus/ctxcdb/dbtest"
	"testing"
)

func TestMemoryDB(t *testing.T) {
	t.Run("DatabaseSuite", func(t *testing.T) {
		dbtest.TestDatabaseSuite(t, func() ctxcdb.KeyValueStore {
			return New()
		})
	})
}

