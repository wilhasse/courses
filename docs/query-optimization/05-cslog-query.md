# cslog-query — The Next Generation

> The newest system: a Rust analytical query accelerator that serves slow MySQL reports
> from a local, immutable analytical copy. **In development** — deliberately starting
> simple and improving slowly, with every step verified.

## A different answer to the same problem

[SmartSQL](./04-smart-sql.md) accelerates queries *against* live MySQL. cslog-query flips
the strategy: **take the analytical workload off MySQL entirely.** It maintains a local,
read-only analytical copy of the hot tables and answers the slow reports itself — MySQL
never sees them.

The lineage matters: this is the disciplined restart of an earlier lab (`cslog-db`, which
grew toward Apache-Doris parity and collapsed under its own generality — see
[Labs](./03-labs.md)), keeping only the ideas that proved out and none of the sprawl.

```
MySQL snapshot + binlog CDC
      │
      ▼
Arrow batches → local immutable rowsets (Parquet) → atomic snapshots
      │
      ▼
Apache DataFusion queries  ──►  served read-only over the MySQL wire protocol
      │
      ▼
optional materialized report tables (precomputed slow reports)
```

**Scope is a feature**: this is *not* a general OLAP product, not a distributed cluster —
it makes specific slow customer reports fast on modest hardware. (~41k lines of Rust,
~6.5k lines of design docs, ticket-driven development.)

## The concepts, from simple upward

- **Immutable rowsets + atomic snapshot flips.** Data files are never modified; a new
  snapshot becomes visible by atomically flipping a `CURRENT` pointer (metadata written
  first, pointer last). Readers always see a consistent snapshot — the same
  copy-on-write/versioning philosophy as InnoDB's MVCC, at file granularity.
- **Replay-safe CDC.** Binlog ingestion is at-least-once with idempotent apply
  (batch/sequence dedup); primary-key tables get upserts/deletes via **roaring-bitmap
  delete vectors**; PK-less tables are append-only. DDL or schema drift on a tracked
  table fails safe (stop, don't corrupt).
- **Pluggable storage engines** — explicitly modeled on MySQL's storage-engine split. A
  clean `RowsetStorageBackend` trait separates the API from two interchangeable backends
  that run **side by side over the same source** for comparison: a simple Parquet
  bootstrap engine and a "DorisRust" engine (segments, sharded PK indexes, zone-map
  pruning, delete bitmaps). Bake-offs, not beliefs.
- **One long-lived DataFusion session** with snapshot-gated table registration — a direct
  fix for the predecessor's per-query session overhead (which cost ~20× on fast queries).
  The new streaming table provider registers in O(metadata) and pushes projections,
  filters, and limits down into the segment scan.
- **Hardcoded fast paths, kept honest by benchmarks.** A few known query shapes (point
  lookups, one hot join, `COUNT(*)`) bypass the planner entirely. The rule is empirical:
  a fast path retires only when the general engine gets within 2× — measured, and so far
  the fast path wins by ~11× over the wire.
- **Parity as a first-class feature.** A normalization + checksum harness compares the
  accelerator's results against MySQL's for the same query — the report is only trusted
  when the checksums match.

## Where it stands

Development runs in ticket-driven phases, each fully verified before the next
(276 tests passing at last count):

1. ✅ **CDC correctness & durability** — consistent snapshot/binlog coordination, fsync
   barriers, type-mapping fixes, fail-safe DDL handling, lag alerting.
2. ✅ **DataFusion integration** — generic SQL fallback, pushdowns, caches, streaming
   provider, shared session, fast-path benchmark gate.
3. 🔄 **First real report offload** — freezing a production slow report (a multi-table,
   multi-hour monster) as a fixture, replicating its 6 tables, running the parity
   harness, then materializing it.

The "start simple, improve slowly" discipline is the point: every layer (ingestion,
storage, query, parity) earns trust before the next is built on top — a deliberate
contrast to the predecessor lab that tried to be Doris on day one.

## The two products, side by side

| | SmartSQL | cslog-query |
|---|---|---|
| strategy | accelerate queries **against** MySQL | move reports **off** MySQL |
| stack | Go + chDB (embedded ClickHouse) | Rust + Arrow/Parquet + DataFusion |
| data | MySQL remains source of truth for results | local analytical copy serves results |
| freshness | binlog CDC into columnar mirror | binlog CDC into immutable rowsets |
| client view | same MySQL, transparently faster | separate read-only MySQL endpoint |
| status | production (curated queries) | development (first report in progress) |

Both descend from the same idea chain — and both speak the MySQL wire protocol, because
the least disruptive database technology is the one your applications can't tell apart
from MySQL.

---
**Previous:** [SmartSQL](./04-smart-sql.md) · **Back to:** [The Journey](./index.md)
