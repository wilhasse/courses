# Article 1 — Why Rewrite InnoDB in Rust?

> The motivation, the method, and what "parity" actually means here.

## Three motivations, one project

1. **The ultimate comprehension test.** Reading source (the
   [InnoDB course](../innodb-architecture/README.md) came from that) teaches you what the
   engine does; only *reimplementing* it teaches you what every byte is for. The exam is
   objective: either MySQL's `IMPORT TABLESPACE` accepts your pages or it doesn't.
2. **Tooling that didn't exist.** The [journey](../../query-optimization/index.md)
   repeatedly needed to work with InnoDB files *outside* a running server — parsing them
   offline (percona-parser), replicating them, one day repairing them. A library that can
   read **and write** the format enables a whole class of tools: `ibd_create`,
   `ibd_insert`, `ibd_scan`, `ibd_validate`, and an `ibd2sdi` clone all fell out of it.
3. **An embedded engine, reborn.** The project exposes the classic **Embedded InnoDB
   `ib_*` API** (~120 functions — a superset of the 2009 original studied in the course).
   The long-game vision: what Innobase Oy shipped in 2009 — a linkable transactional
   storage engine — with modern InnoDB's format and Rust's safety.

## The method: ticket-by-ticket parity porting

The discipline matters as much as the code. Every unit of work follows the same loop:

```
pick a subsystem ticket (IBRUS-NNN, ~450 so far)
  → read the upstream C++ (Percona Server 8.0, storage/innobase/)
  → implement the Rust equivalent (same constants, same offsets, same names)
  → test it (unit + round-trip where applicable)
  → log the ticket, commit, next
```

Two reference codebases anchor the port: **Percona Server 8.0** for the modern on-disk
format and algorithms, and the **2009 Embedded InnoDB** tree for the public API surface.
Work was organized in 13 phases (foundation → page format → space management → B+tree →
dictionary/SDI → transactions → …), echoing the bottom-up layering of the course — you
can't write a B+tree page before you can checksum a page.

"Parity" is used in a precise sense: **same bytes, same constants, same structural
behavior** — `FIL_PAGE_INDEX` is 17855, the compact record header is 5 bytes read at
negative offsets, directory slots own 4-8 records — verified not by assertion but by
making real MySQL consume the output (Article 4).

## Why Rust (briefly)

The usual reasons apply (memory safety in a codebase whose C ancestor is famous for
pointer arithmetic into 16KB byte arrays; fearless refactoring; `cargo test`/fuzz/CI
ergonomics) — but the honest primary answer is: **because the point was to learn deeply,
and translation forces understanding.** Rust's ownership model also has an interesting
side effect: InnoDB idioms that rely on shared mutable state (latch-protected page frames,
the kernel mutex) can't be transliterated — you must first understand *why* the original
needs them, then design an equivalent. Every fight with the borrow checker was a lesson
about InnoDB's concurrency model.

## What it is not

- Not a MySQL server plugin — engine-only, no `handler/` glue.
- Not (yet) crash-safe or concurrent — Article 5 is explicit about the durability and
  locking gaps; today it is a **format-complete, single-threaded engine + offline
  tooling**, not a production database.
- Not a fork or binding — no InnoDB C code is linked; every byte is produced by Rust.

---
**Next:** [Architecture: 23 Crates Mirroring InnoDB](./02-architecture.md)
