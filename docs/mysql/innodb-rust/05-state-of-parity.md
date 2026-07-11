# Article 5 — The State of Parity: An Honest Assessment

> What matches upstream InnoDB, what doesn't yet, and the roadmap — written the way a
> real engineering assessment should be: gaps first, no varnish.

The project maintains a systematic parity report against its upstream baseline
(Percona Server 8.0.46, ~500k lines of C/C++), produced by per-subsystem source
comparison — checking not just "does a module exist" but whether the *mechanisms*
match, down to symbol-level grep for things that should exist and don't.

## What has strong parity ✅

The **on-disk formats and single-threaded logic** — which is exactly what the
read/write/round-trip goal required:

- Page layouts and both page checksums (CRC32C + legacy); COMPACT records including
  instant-DDL row versioning; B-tree search/insert/split/root-raise.
- **Compressed pages** (`page0zip`) with real zlib + LZ4 and the modification log.
- SDI encode/decode plus `.cfg` transport metadata — the full IMPORT/EXPORT dance.
- MVCC read views and isolation levels; the classic lock-mode compatibility matrix.
- LOB insert/read in the 8.0 layout; buffer pool LRU with dump/restore.
- A 5.7-style `SYS_*` data dictionary; clustered + secondary DML with FK cascades.
- **~120 `ib_*` embedded-API functions** — a superset of the historical Embedded InnoDB
  surface (the same API documented in the [InnoDB course](../innodb-architecture/README.md)).

## The three headline gaps ⚠️

The assessment names three correctness-blocking areas — notably, all three are things
the [InnoDB course](../innodb-architecture/README.md) identifies as the *hard 20%* of a
storage engine:

1. **The crash-durability chain is not real yet.** Redo block checksums are stubbed;
   only a handful of MLOG record types parse; undo logs live in memory with no
   `DB_ROLL_PTR` threading; recovery rolls back every transaction instead of resurrecting
   committed ones; the doublewrite buffer has no recovery half. The write-ahead-logging
   *format* exists — the *guarantee* doesn't.
2. **Concurrency is decorative.** Latch modes are accepted and ignored; scans acquire no
   record locks (`SELECT ... FOR UPDATE` and next-key phantom protection are absent);
   there are no background threads (master/purge/IO). Single-threaded correctness only.
3. **No file-segment allocation (`fseg_*`).** Called out as the single most critical
   gap: indexes bypass segment accounting, so trees cannot grow and shrink the way real
   tablespaces demand.

Also missing: collation-aware record comparison (byte-order only today), REDUNDANT-format
user records, the 8.0 transactional dictionary, async I/O, FTS persistence — plus one
known open bug on certain datetime + nullable-varchar row layouts.

## Why this list is a feature, not an embarrassment

Two things make this assessment worth publishing:

- **It's the mirror image of the course.** Everything with strong parity is Layers 1-2
  (and slices of 4-5) of the InnoDB deep-dive's model — formats, pages, trees. Everything
  in the gap list is Layer 3 (durability) and the cross-cutting concurrency machinery.
  Reimplementing InnoDB teaches you *precisely* where the engineering weight sits: the
  formats took months; the guarantees are the years.
- **The roadmap is dependency-ordered**, not wishful: segment allocation first (trees
  must grow), then real redo (CRC32 + full MLOG catalog + apply), then durable undo with
  `roll_ptr` and transaction resurrection, then doublewrite recovery, then record locking
  in scans, then collation compare, then the background-thread tail. Each step unlocks
  the next — the same bottom-up ordering the course uses to *teach* the engine, now used
  to *build* it.

## Method notes

Development is ticket-driven (450+ tickets over ~6 months, 558 commits), with each
ticket following the same loop: read the upstream C++ mechanism → implement the Rust
equivalent → test → document. The per-phase plans and per-ticket logs make the repo an
unusually complete record of *how* to port a large C++ system — which may end up being
the project's most valuable output, whatever the code becomes.

---
**Previous:** [Parity Testing](./04-parity-testing.md) · **Back to:** [Series Overview](./README.md)
