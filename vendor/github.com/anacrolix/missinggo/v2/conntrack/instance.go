package conntrack

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net"
	"sort"
	"strconv"
	"text/tabwriter"
	"time"

	"github.com/anacrolix/stm"
	"github.com/anacrolix/stm/stmutil"

	"github.com/anacrolix/missinggo/v2"
	"github.com/anacrolix/missinggo/v2/iter"
)

type reason = string

type Instance struct {
	maxEntries   *stm.Var
	noMaxEntries *stm.Var
	Timeout      func(Entry) time.Duration

	// Occupied slots
	entries *stm.Var

	// priority to entryHandleSet, ordered by priority ascending
	waitersByPriority *stm.Var //Mappish
	waitersByReason   *stm.Var //Mappish
	waitersByEntry    *stm.Var //Mappish
	waiters           *stm.Var // Settish
}

type (
	priority int
)

func NewInstance() *Instance {
	i := &Instance{
		// A quarter of the commonly quoted absolute max on a Linux system.
		maxEntries:   stm.NewVar(1 << 14),
		noMaxEntries: stm.NewVar(false),
		Timeout: func(e Entry) time.Duration {
			// udp is the main offender, and the default is allegedly 30s.
			return 30 * time.Second
		},
		entries: stm.NewVar(stmutil.NewMap()),
		waitersByPriority: stm.NewVar(stmutil.NewSortedMap(func(l, r interface{}) bool {
			return l.(priority) > r.(priority)
		})),

		waitersByReason: stm.NewVar(stmutil.NewMap()),
		waitersByEntry:  stm.NewVar(stmutil.NewMap()),
		waiters:         stm.NewVar(stmutil.NewSet()),
	}
	return i
}

func (i *Instance) SetNoMaxEntries() {
	stm.AtomicSet(i.noMaxEntries, true)
}

func (i *Instance) SetMaxEntries(max int) {
	stm.Atomically(stm.VoidOperation(func(tx *stm.Tx) {
		tx.Set(i.noMaxEntries, false)
		tx.Set(i.maxEntries, max)
	}))
}

func (i *Instance) remove(eh *EntryHandle) {
	stm.Atomically(func(tx *stm.Tx) interface{} {
		es, _ := deleteFromMapToSet(tx.Get(i.entries).(stmutil.Mappish), eh.e, eh)
		tx.Set(i.entries, es)
		return nil
	})
}

func deleteFromMapToSet(m stmutil.Mappish, mapKey, setElem interface{}) (stmutil.Mappish, bool) {
	_s, ok := m.Get(mapKey)
	if !ok {
		return m, true
	}
	s := _s.(stmutil.Settish)
	s = s.Delete(setElem)
	if s.Len() == 0 {
		return m.Delete(mapKey), true
	}
	return m.Set(mapKey, s), false
}

func (i *Instance) deleteWaiter(eh *EntryHandle, tx *stm.Tx) {
	tx.Set(i.waiters, tx.Get(i.waiters).(stmutil.Settish).Delete(eh))
	tx.Set(i.waitersByPriority, stmutil.GetLeft(deleteFromMapToSet(tx.Get(i.waitersByPriority).(stmutil.Mappish), eh.priority, eh)))
	tx.Set(i.waitersByReason, stmutil.GetLeft(deleteFromMapToSet(tx.Get(i.waitersByReason).(stmutil.Mappish), eh.reason, eh)))
	tx.Set(i.waitersByEntry, stmutil.GetLeft(deleteFromMapToSet(tx.Get(i.waitersByEntry).(stmutil.Mappish), eh.e, eh)))
}

func (i *Instance) addWaiter(eh *EntryHandle) {
	stm.Atomically(stm.VoidOperation(func(tx *stm.Tx) {
		tx.Set(i.waitersByPriority, addToMapToSet(tx.Get(i.waitersByPriority).(stmutil.Mappish), eh.priority, eh))
		tx.Set(i.waitersByReason, addToMapToSet(tx.Get(i.waitersByReason).(stmutil.Mappish), eh.reason, eh))
		tx.Set(i.waitersByEntry, addToMapToSet(tx.Get(i.waitersByEntry).(stmutil.Mappish), eh.e, eh))
		tx.Set(i.waiters, tx.Get(i.waiters).(stmutil.Settish).Add(eh))
	}))
}

func addToMapToSet(m stmutil.Mappish, mapKey, setElem interface{}) stmutil.Mappish {
	s, ok := m.Get(mapKey)
	if ok {
		s = s.(stmutil.Settish).Add(setElem)
	} else {
		s = stmutil.NewSet().Add(setElem)
	}
	return m.Set(mapKey, s)
}

func (i *Instance) WaitDefault(ctx context.Context, e Entry) *EntryHandle {
	return i.Wait(ctx, e, "", 0)
}

// Nil returns are due to context completion.
func (i *Instance) Wait(ctx context.Context, e Entry, reason string, p priority) (eh *EntryHandle) {
	eh = &EntryHandle{
		reason:   reason,
		e:        e,
		i:        i,
		priority: p,
		created:  time.Now(),
	}
	i.addWaiter(eh)
	ctxDone, cancel := stmutil.ContextDoneVar(ctx)
	defer cancel()
	success := stm.Atomically(func(tx *stm.Tx) interface{} {
		es := tx.Get(i.entries).(stmutil.Mappish)
		if s, ok := es.Get(e); ok {
			tx.Set(i.entries, es.Set(e, s.(stmutil.Settish).Add(eh)))
			return true
		}
		haveRoom := tx.Get(i.noMaxEntries).(bool) || es.Len() < tx.Get(i.maxEntries).(int)
		topPrio, ok := iter.First(tx.Get(i.waitersByPriority).(iter.Iterable).Iter)
		if !ok {
			panic("y u no waiting")
		}
		if haveRoom && p == topPrio {
			tx.Set(i.entries, addToMapToSet(es, e, eh))
			return true
		}
		if tx.Get(ctxDone).(bool) {
			return false
		}
		tx.Retry()
		panic("unreachable")
	}).(bool)
	stm.Atomically(stm.VoidOperation(func(tx *stm.Tx) {
		i.deleteWaiter(eh, tx)
	}))
	if !success {
		eh = nil
	}
	return
}

func (i *Instance) Allow(tx *stm.Tx, e Entry, reason string, p priority) *EntryHandle {
	eh := &EntryHandle{
		reason:   reason,
		e:        e,
		i:        i,
		priority: p,
		created:  time.Now(),
	}
	es := tx.Get(i.entries).(stmutil.Mappish)
	if s, ok := es.Get(e); ok {
		tx.Set(i.entries, es.Set(e, s.(stmutil.Settish).Add(eh)))
		return eh
	}
	haveRoom := tx.Get(i.noMaxEntries).(bool) || es.Len() < tx.Get(i.maxEntries).(int)
	topPrio, ok := iter.First(tx.Get(i.waitersByPriority).(iter.Iterable).Iter)
	if haveRoom && (!ok || p == topPrio) {
		tx.Set(i.entries, addToMapToSet(es, e, eh))
		return eh
	}
	return nil
}

func parseHostPort(hostport string) (ret struct {
	hostportErr error
	host        string
	hostIp      net.IP
	port        string
	portInt64   int64
	portIntErr  error
}) {
	ret.host, ret.port, ret.hostportErr = net.SplitHostPort(hostport)
	ret.hostIp = net.ParseIP(ret.host)
	ret.portInt64, ret.portIntErr = strconv.ParseInt(ret.port, 0, 64)
	return
}

func (i *Instance) PrintStatus(w io.Writer) {
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "num entries: %d\n", stm.AtomicGet(i.entries).(stmutil.Lenner).Len())
	fmt.Fprintln(w)
	fmt.Fprintf(w, "%d waiters:\n", stm.AtomicGet(i.waiters).(stmutil.Lenner).Len())
	fmt.Fprintf(tw, "num\treason\n")
	stm.AtomicGet(i.waitersByReason).(stmutil.Mappish).Range(func(r, ws interface{}) bool {
		fmt.Fprintf(tw, "%d\t%q\n", ws.(stmutil.Settish).Len(), r.(reason))
		return true
	})
	tw.Flush()
	fmt.Fprintln(w)
	fmt.Fprintln(w, "handles:")
	fmt.Fprintf(tw, "protocol\tlocal\tremote\treason\texpires\tcreated\n")
	entries := stm.AtomicGet(i.entries).(stmutil.Mappish)
	type entriesItem struct {
		Entry
		stmutil.Settish
	}
	entriesItems := make([]entriesItem, 0, entries.Len())
	entries.Range(func(e, hs interface{}) bool {
		entriesItems = append(entriesItems, entriesItem{e.(Entry), hs.(stmutil.Settish)})
		return true
	})
	sort.Slice(entriesItems, func(i, j int) bool {
		l := entriesItems[i].Entry
		r := entriesItems[j].Entry
		var ml missinggo.MultiLess
		f := func(l, r string) {
			pl := parseHostPort(l)
			pr := parseHostPort(r)
			ml.NextBool(pl.hostportErr != nil, pr.hostportErr != nil)
			ml.NextBool(pl.hostIp.To4() == nil, pr.hostIp.To4() == nil)
			ml.Compare(bytes.Compare(pl.hostIp, pr.hostIp))
			ml.NextBool(pl.portIntErr != nil, pr.portIntErr != nil)
			ml.StrictNext(pl.portInt64 == pr.portInt64, pl.portInt64 < pr.portInt64)
			ml.StrictNext(pl.port == pr.port, pl.port < pr.port)
		}
		f(l.RemoteAddr, r.RemoteAddr)
		ml.StrictNext(l.Protocol == r.Protocol, l.Protocol < r.Protocol)
		f(l.LocalAddr, r.LocalAddr)
		return ml.Less()
	})
	for _, ei := range entriesItems {
		e := ei.Entry
		ei.Settish.Range(func(_h interface{}) bool {
			h := _h.(*EntryHandle)
			fmt.Fprintf(tw,
				"%q\t%q\t%q\t%q\t%s\t%v ago\n",
				e.Protocol, e.LocalAddr, e.RemoteAddr, h.reason,
				func() interface{} {
					if h.expires.IsZero() {
						return "not done"
					} else {
						return time.Until(h.expires)
					}
				}(),
				time.Since(h.created),
			)
			return true
		})
	}
	tw.Flush()
}
