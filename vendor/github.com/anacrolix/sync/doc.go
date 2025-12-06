// Package sync is an extension of the stdlib "sync" package. It has extra functionality that helps
// debug the use of synchronization primitives. The package should be importable in place of "sync".
// The extra functionality can be enabled by calling Enable() or passing a non-empty PPROF_SYNC
// environment variable to the process.
//
// Several profiles are exposed on the default HTTP muxer (and to "/debug/pprof" when
// "net/http/pprof" is imported by the process). "lockHolders" lists the stack traces of goroutines
// that called Mutex.Lock that haven't subsequently been Unlocked. "lockBlockers" contains
// goroutines that are waiting to obtain locks. "/debug/lockTimes" or PrintLockTimes() shows the
// longest time a lock is held for each stack trace.
//
// Disable any overhead and make env and functions no-ops by setting disable_pprof_sync tag.
package sync
