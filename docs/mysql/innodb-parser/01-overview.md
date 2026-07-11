# Article 1 — Reading InnoDB Without a Server

> What the tool does, why offline reading is valuable, and where it sits in the journey.

## The problem it solves

An InnoDB `.ibd` file at rest is not readable by casual tools once two production
features are turned on:

- **Encryption at rest** — the pages are AES-encrypted; you need the keyring master key
  and the tablespace's wrapped key to see anything.
- **Compression** (`ROW_FORMAT=COMPRESSED`) — index pages are zlib page-zip blobs that
  must be decompressed with InnoDB's exact algorithm to reconstruct the 16KB page.

innodb-parser reads such files **offline** — no running MySQL, no crash recovery, no
server process. Point it at a single-table `.ibd` (plus the schema, extracted once with
`ibd2sdi`) and it will decrypt it, decompress it, walk the B+tree leaf pages, and stream
the rows out as pipe-delimited / CSV / JSONL — decoding LOBs, binary JSON, charset-aware
text, and temporal types along the way.

## Why offline matters

This capability is not academic — it's the pragmatic escape hatch the
[journey](../../query-optimization/index.md) kept reaching for:

- **Replication's blind spots.** MySQL's binlog misses some changes (notably
  [FK-cascade deletes](../../query-optimization/01-the-idea.md)). Reading the *files*
  sees the true state, whatever the binlog said.
- **Recovery & forensics.** A dead server, a dropped table, a corrupt instance — the
  data is still in the `.ibd`; a file reader gets it out.
- **Feeding analytical systems.** Bulk-loading a columnar mirror from files is faster and
  simpler than replaying a change stream — and the C ABI (Article 4) made this reusable
  from Rust and Go.

## Capabilities at a glance

| capability | detail |
|-----------|--------|
| Decrypt | Percona keyring master key → AES-256 unwrap of tablespace key/IV → AES-256-CBC page decryption |
| Decompress | page-zip (zlib) index pages via InnoDB's own `page_zip_decompress_low`, 8KB→16KB |
| Parse | COMPACT **and** REDUNDANT records; clustered or secondary index; custom offset handling |
| Decode | LOB/ZLOB external pages, MySQL binary JSON, charset-aware strings, DATETIME/DECIMAL/… |
| Rebuild (experimental) | reconstruct uncompressed 16KB pages, rebuild SDI, remap index IDs, emit `.cfg` for `IMPORT TABLESPACE` |
| Output | pipe / CSV / JSONL, optional per-row metadata (page, offset, delete flag) |
| Integrate | a C ABI (`libibd_reader.so`) for FFI from Go/Python/Rust |

Five CLI modes make each stage independently usable: `1` decrypt-only, `2`
decompress-only, `3` parse, `4` decrypt+decompress in one pass, `5` rebuild-for-import.

## Where it sits in the journey

This is the middle rung of a three-step relationship with InnoDB's format:

```
study the source        →   bend the source into a tool   →   rebuild it from scratch
InnoDB Architecture          innodb-parser (C, this series)     InnoDB in Rust
(read the code)              (link + stub real MySQL code)       (reimplement in Rust)
```

Each rung is a deeper level of ownership. Reading source teaches you the format;
compiling that source into a standalone tool forces you to understand its *dependencies*
(what does the decompressor actually need? what globals does the keyring touch?);
rewriting it in Rust forces you to understand every *byte*. innodb-parser is where the
abstract knowledge first became an executable that touches real, production-encrypted
files.

## Honest limitations

- Needs an SDI JSON schema (from `ibd2sdi`) — it reads data, not DDL.
- Single-table `.ibd` only (not the shared `ibdata1`, undo, or temp tablespaces).
- MySQL 8+ format; one index per run; single-threaded.
- Rebuilt files retain COMPRESSED metadata, so re-`IMPORT` is experimental (the tool
  demonstrates the expected mismatch rather than hiding it).
- Transparent *page* compression (whole-page punch-hole) is detected but not expanded.

---
**Next:** [The Build Challenge](./02-build-challenge.md)
