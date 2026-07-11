# Article 2 — Architecture: 23 Crates Mirroring InnoDB

> The workspace is a deliberate map of upstream `storage/innobase/` — if you know the
> C++ tree (or the [course](../innodb-architecture/README.md)), you know where everything
> lives.

## One crate per subsystem

The Cargo workspace mirrors InnoDB's directory-per-subsystem layout — the same `xxx0yyy`
modules the course walked through, reborn as crates:

| crate | upstream twin | role (≈ LOC) |
|-------|---------------|--------------|
| `innodb-core` | `ut0*`, `mach0data` | errors, big-endian `mach_read/write_*`, CRC32C (1.6k) |
| `innodb-page` | `page0*`, `fil0fil` (page part) | FIL header/trailer, index page header, page directory, record format, checksums, page0zip compression (4k) |
| `innodb-fil` / `innodb-fsp` | `fil0*` / `fsp0*` | tablespace files, encryption (AES-CBC/CTR); FSP header, XDES, inodes (3.6k) |
| `innodb-btr` | `btr0*` | search, cursors + persistent cursors, page insert/delete, splits, root raise, node pointers, bulk load, validation (6.7k) |
| `innodb-dict` | `dict0*` + SDI | table defs, data types, **SDI encode/decode**, `.cfg` transport (9.6k) |
| `innodb-row` | `row0*` | insert/search/update, FK cascade, LOB, undo hooks (11k) |
| `innodb-trx` / `innodb-lock` | `trx0*` / `lock0*` | trx lifecycle, read views, rsegs, purge, XA; lock-mode matrix, gap/next-key locks (9.7k) |
| `innodb-log` / `innodb-buf` | `log0*` / `buf0*` | mtr, LSN/blocks, checkpoint, recovery scaffold; buffer pool, LRU, doublewrite writer, flush (5k) |
| `innodb-ddl`, `innodb-ahi`, `innodb-ibuf`, `innodb-fts`, `innodb-gis`, `innodb-clone` | their namesakes | DDL/instant/online, adaptive hash, change buffer, full-text, R-tree, clone (8k) |
| `innodb-api` | `api0api` (Embedded InnoDB) | the `ib_*` public API: startup, schema, cursors, tuples (7.8k) |
| `innodb-sql`, `innodb-demo`, `demo-table` | — | sqlparser front, MySQL-protocol demo server + E2E suite, live-table replicator (8.5k) |
| `innodb-test` | — | the CLI tools: `ibd_create`, `ibd_insert`, `ibd_scan`, `ibd_validate`, `ibd2sdi-rust`, regression/bench harness (11.5k) |
| `innodb` | — | facade crate, feature-gated re-exports |

Three architectural choices worth noting:

- **Hand-rolled byte codecs, no parser framework.** Everything reads/writes through
  `mach_read_from_4`-style helpers — the same primitives as upstream. For a format
  defined in C by byte offsets, a declarative parser layer would only add distance from
  the reference. (Dependencies stay minimal: zlib/LZ4/zstd, AES, serde for SDI JSON,
  sqlparser + opensrv-mysql for the demo stack.)
- **Traits where upstream has globals.** Page access goes through a `BtrPageProvider`
  trait rather than a global buffer pool — so the B+tree code runs identically against
  an in-memory map (tests), the buffer pool crate, or a raw file (offline tools). This
  single inversion is what makes `ibd_insert` — *offline* B+tree surgery on a closed
  tablespace — possible.
- **The API crate is the roof.** Everything converges on `innodb-api`'s `ib_*` surface:
  `ib_startup`, `ib_cursor_open_table`, `ib_cursor_insert_row`… — the 2009 embedded API,
  now backed by 8.0-format internals. The demo server (opensrv-mysql + sqlparser) then
  puts a MySQL wire protocol in front of it, closing the circle: a queryable,
  MySQL-speaking, pure-Rust InnoDB.

## Layering

The dependency graph enforces the same bottom-up order the course teaches:

```
core → page → {fil, fsp} → btr → dict → {row, trx, lock, log, buf} → ddl/… → api → demo
```

No crate reaches "up"; the B+tree doesn't know about transactions, pages don't know about
trees. When a layering violation is tempting, that's usually the signal that upstream
solves it with a callback or shared struct — and a design decision has to be made rather
than transliterated.

---
**Previous:** [Why Rewrite InnoDB in Rust?](./01-why-rust-innodb.md) · **Next:** [Reading & Writing the Format](./03-reading-writing-ibd.md)
