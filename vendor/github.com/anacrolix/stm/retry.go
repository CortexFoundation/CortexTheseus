package stm

// Retry is a sentinel value. When thrown via panic, it indicates that a
// transaction should be retried.
const Retry = "retry"

// catchRetry returns true if fn calls tx.Retry.
func catchRetry(fn func(*Tx), tx *Tx) (retry bool) {
	defer func() {
		if r := recover(); r == Retry {
			retry = true
		} else if r != nil {
			panic(r)
		}
	}()
	fn(tx)
	return
}
