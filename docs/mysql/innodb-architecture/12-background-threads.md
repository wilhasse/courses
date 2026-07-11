# Chapter 12 — Background Threads, the Master Thread & the Insert Buffer

> Who does the work your transaction didn't wait for — deferred flushing, purging, merging,
> checkpointing.
> Source: `srv/srv0srv.c`, `srv/srv0start.c`, `ibuf/ibuf0ibuf.c`

## 12.1 The thread roster

A running engine has surprisingly few threads (`srv0start.c:1457-1777`):

| thread | role |
|--------|------|
| N × `io_handler_thread` (`:593`) | complete simulated-async I/O requests (Ch. 1) — segments for log, ibuf, reads, writes |
| `srv_master_thread` (`srv0srv.c:2215`) | *the* background worker — below |
| `srv_lock_timeout_thread` (`srv0srv.c:1961`) | wakes lock waiters whose timeout expired (Ch. 8) |
| `srv_error_monitor_thread` (`:2059`) | detects long semaphore waits (latch bugs/stalls, Ch. 4) |
| `srv_monitor_thread` (`:1754`) | prints the InnoDB monitor output |
| user threads | your application's threads executing `ib_*` calls |

Note what's *not* here: no dedicated purge threads, no page-cleaner threads, no log-writer
thread — all of that lives inside **one master thread** in this era. Watching later InnoDB
versions peel these responsibilities into dedicated threads is watching the scalability story
of the 2010s.

## 12.2 The master thread: the engine's heartbeat

`srv_master_thread()` (`srv/srv0srv.c:2215`) is a loop with two cadences — a translation of
the classic "spend spare I/O capacity wisely" policy into code:

```
every second (srv0srv.c:2280-2384):
  • flush the log buffer                    ← makes flush_log_at_trx_commit=0/2 safe-ish
  • if I/O was idle:  merge insert buffer   (5% of I/O capacity)
  • if buf pool too dirty (> max_dirty_pages_pct):
        buf_flush_batch(BUF_FLUSH_LIST, 100% of I/O capacity)
  • adaptive flushing based on redo production rate

every ten seconds (:2386-2464):
  • flush 100 dirty pages if I/O was idle
  • ALWAYS merge some insert buffer
  • trx_purge() until history drained       ← MVCC garbage collection (Ch. 7)
  • flush oldest pages (100 if >70% dirty, else 10)
  • log_checkpoint()                        ← bound recovery time (Ch. 5)

when the server is idle: background loop — purge, ibuf merge, big flushes
```

Every decision is gated on measured I/O activity (`PCT_IO(n)` = percent of the configured
`io_capacity`, thresholds at `srv0srv.c:406-408`): background work must not steal bandwidth
from user transactions, but idle disks should be earning their keep. That single idea —
*deferred work + idle-time scheduling* — is why InnoDB commits fast: a transaction only pays
for redo-log durability at commit; page writes, purging, ibuf merging, and checkpointing are
all somebody else's (this thread's) problem.

## 12.3 The insert buffer: turning random I/O into sequential

The last major structure in the engine, and this era's signature optimization
(`ibuf/ibuf0ibuf.c`).

**Problem.** Every row INSERT updates each secondary index. Secondary-index keys are
essentially random (that's why you built the index), so each insert wants a *random leaf
page* — if it's not cached, that's a random disk read *per index per row*. Bulk loads die by
this.

**Solution.** If the target leaf page of a **non-unique secondary index** is not in the buffer
pool (`btr_cur_search_to_nth_level` probes with `BUF_GET_IF_IN_POOL`,
`btr/btr0cur.c:537-544`), don't read it. Instead, insert the entry into the **insert buffer
tree** — a system B+tree (space 0, root at page 4, Ch. 1) keyed by
(space, page_no, record) — and mark completion (`ibuf_insert`, `ibuf/ibuf0ibuf.c:2842`). The
random read is avoided entirely.

**Merging.** Buffered entries are applied when the page comes into the pool anyway:

- on a normal read of the page (`ibuf_merge_or_delete_for_page`, `:3150`), or
- proactively by the master thread (`ibuf_contract_for_n_pages`, `:2282`), which reads target
  pages in batches (turning many random writes into few clustered ones).

**Bookkeeping.** How do you know a page has pending entries without reading it? The **ibuf
bitmap pages** (Ch. 1: page 1 of every 16384-page block): 4 bits per page — free-space
estimate, a "has buffered entries" bit, and an "is an ibuf page" bit
(`IBUF_BITMAP_FREE/BUFFERED/IBUF`, `ibuf0ibuf.c:214-218`). The bitmap is consulted on every
buffered insert and cleared on merge.

**The fine print** — why only *non-unique* secondary indexes: a UNIQUE check must read the
actual leaf page to detect duplicates, which defeats the whole point; and clustered-index
pages are where the row itself goes, so they're needed anyway (asserted at
`ibuf0ibuf.c:2858`). There's also a subtle crash-safety story: ibuf entries and their merges
are all redo-logged B+tree operations, so recovery replays them like everything else — but
recovery must *not* trigger new merges while the redo scan runs
(`recv_no_ibuf_operations`, Ch. 5).

Modern MySQL renamed this the "change buffer" (buffering deletes and purges too) — and
MySQL 8.4 finally disabled it by default in 2024, because SSDs made random reads cheap. You
are reading the code of an optimization across its entire life cycle: invented for spinning
disks, obsoleted by flash.

## 12.4 The full picture

With the insert buffer, the engine diagram from Chapter 0 is complete. One last mental replay —
a single-row INSERT touches, in order:

```
api (Ch.11) → row_ins (Ch.9) → dict lookup (Ch.10) → btr descent (Ch.6)
  → [leaf cached? else ibuf (Ch.12)] → page_cur insert (Ch.2)
  → inside an mtr: latches (Ch.4) + redo (Ch.5) → undo record (Ch.7)
  → implicit lock (Ch.8) → dirty page in buffer pool (Ch.3)
commit: undo state flip + log group-commit (Ch.7, 5)
later, invisibly: master thread flushes the page via doublewrite (Ch.3, 12),
  checkpoint advances (Ch.5), purge/ibuf-merge tidy up (Ch.7, 12)
crash at ANY point: recovery redoes, then undoes (Ch.5) — no step above is lost.
```

If you can narrate that chain from memory, you understand how InnoDB was architected.

## 12.5 What to remember

1. One master thread runs the engine's metabolism on 1s/10s cadences, spending only idle I/O —
   deferred work is the reason commits are fast.
2. The insert buffer trades "random read now" for "sequential merge later", tracked by bitmap
   pages; it's the era's boldest optimization and a lesson in hardware-driven design.
3. Timeouts, monitors, and I/O completion each get a small dedicated thread; everything else
   is done by whoever's transaction needs it — a design later versions had to unbundle to
   scale on many-core machines.

**Try it:** run `tests/run_bulk_test.sh 100000 1000 4`, then break in `srv_master_thread` —
step through one 10-second cycle and watch it call `ibuf_contract_for_n_pages`, `trx_purge`,
`buf_flush_batch`, and `log_checkpoint` in turn.

---
**Previous:** [Chapter 11 — Startup & API](./11-startup-api.md) · **Back to:** [Chapter 0 — Overview](./00-overview.md)
