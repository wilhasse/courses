# InnoDB in Rust

A 5-article series documenting **innodb-rust**: a from-scratch Rust reimplementation of
the InnoDB storage engine that can **read and write the real InnoDB on-disk format** —
verified by round-tripping tablespaces through an actual MySQL server.

This is the practical companion to the
[InnoDB Architecture Deep-Dive](../innodb-architecture/README.md): that series *explains*
the engine from source; this one documents what happens when you **rebuild it** — what's
easy, what's brutally hard, and how to prove your bytes are right.

## Headline facts

- ~96k lines of Rust across 23 crates, each mirroring an upstream `storage/innobase/`
  subdirectory; tracking **Percona Server 8.0** as the reference, exposing the classic
  **Embedded InnoDB `ib_*` API** (the very API of the deep-dive course's codebase).
- Creates valid `.ibd` tablespaces **from scratch** — FSP headers, SDI dictionary
  records, B+tree pages — that a real MySQL accepts via `IMPORT TABLESPACE`.
- **Bidirectional round-trips**: MySQL writes → Rust reads & modifies → MySQL imports and
  reads back, multiple passes, field-by-field verified.
- Ticket-driven port (450+ tickets, ~6 months), with a systematic, brutally honest
  parity report against upstream.

## The articles

| # | Article | What it covers |
|---|---------|----------------|
| 1 | [Why Rewrite InnoDB in Rust?](./01-why-rust-innodb.md) | The motivation, the method, and what "parity" means |
| 2 | [Architecture: 23 Crates Mirroring InnoDB](./02-architecture.md) | The workspace design and how it maps to upstream |
| 3 | [Reading & Writing the On-Disk Format](./03-reading-writing-ibd.md) | The read path, and the differentiator: building tablespaces byte-by-byte |
| 4 | [Proving It: Parity Testing](./04-parity-testing.md) | Round-trips through real MySQL, SDI parity, fuzzing, crash soaks |
| 5 | [The State of Parity](./05-state-of-parity.md) | What matches, the three hard gaps, and the dependency-ordered roadmap |

## Where this fits in the bigger story

The project grew out of the [query optimization journey](../../query-optimization/index.md)'s
study thread: after parsing `.ibd` files offline (percona-parser), studying the Embedded
InnoDB codebase, and writing the architecture courses, reimplementation was the final
comprehension test — *you understand a storage engine when your implementation of its
format survives the original's import checks.*
