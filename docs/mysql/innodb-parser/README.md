# InnoDB Parser (C)

A 4-article series on **innodb-parser**: a C/C++ toolkit that reads MySQL 8 InnoDB
`.ibd` files *offline* — including **encrypted** and **compressed** tablespaces — by
reusing actual Percona Server source code. It is the low-level, hands-dirty middle of the
journey: after *studying* MySQL's source and before *reimplementing* it in Rust, this is
where you **bend the real source into a standalone tool**.

Where the [InnoDB course](../innodb-architecture/README.md) reads the code and
[InnoDB in Rust](../innodb-rust/README.md) rewrites it, this project does the pragmatic
thing in between: **link MySQL's own C code, stub out the server around it, and make it
run without a database.**

## Why it earns its own series

- It is the only project here written in **C/C++**, compiled *against the Percona Server
  source tree* — a genuine build-engineering fight (dependency dragging, server-global
  stubbing, plugin code coaxed to compile as a library).
- It handles the two things naive `.ibd` readers can't: **transparent decryption**
  (Percona keyring master key → AES-unwrapped tablespace key) and **page-zip
  decompression** (using InnoDB's own decompressor).
- It exposes a **C ABI** that became the FFI backbone other projects in the journey
  consumed to read InnoDB files (e.g. the snapshot-load path of the analytical replicas).

## The articles

| # | Article | What it covers |
|---|---------|----------------|
| 1 | [Reading InnoDB Without a Server](./01-overview.md) | Purpose, capabilities, and where it sits in the journey |
| 2 | [The Build Challenge: Linking MySQL's Own Code](./02-build-challenge.md) | Compiling Percona source offline — stubs, ifdefs, and prebuilt libs |
| 3 | [Decrypt → Decompress → Parse](./03-pipeline.md) | The three-stage pipeline, byte by byte |
| 4 | [A C ABI for Everything Else](./04-c-api-and-verification.md) | The FFI surface and how correctness is proven against real MySQL |

## Journey context

This grew directly out of the [query optimization journey](../../query-optimization/index.md)'s
"parse InnoDB directly" strategy. Reading data files offline sidesteps the
[FK-cascade replication trap](../../query-optimization/01-the-idea.md); the C ABI here is
what let higher-level (Rust, Go) tools ingest InnoDB data without a running MySQL.
