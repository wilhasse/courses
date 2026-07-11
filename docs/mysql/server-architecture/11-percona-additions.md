# Chapter 11 — What Percona Adds

> The delta between Percona Server 8.0 and upstream MySQL 8.0 — and what it teaches about
> the architecture's extension points.
> Source: `sql/threadpool_*.cc`, `storage/rocksdb/`, `plugin/`, `components/`,
> `sql/userstat.cc`

Percona Server is upstream MySQL plus patches that mostly slot into the extension points
this series has mapped: an alternative *connection scheduler*, an alternative *storage
engine*, extra *plugins/components*, and deeper *observability*. Reading the delta is a tour
of where the architecture bends.

## 11.1 Thread pool (a Connection_handler)

Upstream reserves the thread pool for MySQL Enterprise; Percona ships one in the tree
(`sql/threadpool_unix.cc`, activated as the `thread_pool` plugin). It plugs into the exact
seam from Chapter 1 — `Connection_handler_manager::load_connection_handler()` — replacing
thread-per-connection with **thread groups**: connections are sharded across groups; each
group has an epoll-based listener and a small elastic set of workers
(`worker_main`, `threadpool_unix.cc:194`); a timer thread detects stalls and spawns relief
workers. Result: ~constant threads under thousands of connections, no scheduler collapse
under connection storms. Knobs: `thread_pool_size`, `thread_pool_oversubscribe`,
`thread_pool_idle_timeout`.

## 11.2 MyRocks (a handler)

`storage/rocksdb/` (`ha_rocksdb.cc`, ~17.6k lines) wraps Facebook's RocksDB — an
**LSM-tree** engine — behind the Chapter 6 interface. Where InnoDB updates B+tree pages in
place (with redo/undo), an LSM engine turns every write into an append: memtable → sorted
SST files → background compaction. The translation layer is instructive:

- `Rdb_key_def` (`rdb_datadic.h:270`) encodes index entries as **memcomparable keys**
  (byte order = logical order) so RocksDB's plain byte-sorted iterators implement SQL index
  scans — the packing/collation logic InnoDB does per-page comparison, done once at encode
  time.
- The PK is clustered (like InnoDB); secondary keys map to separate **column families**;
  engine metadata lives in a RocksDB system column family (`Rdb_dict_manager`) — the
  "dictionary stored in itself" pattern for the third time in these series.
- Trade-off vs InnoDB: much lower write amplification and better compression (great for
  write-heavy, SSD-bound, replica fleets) versus costlier reads/range scans and a weaker
  fit for mixed workloads — which is why it ships as an *option*, selectable per table with
  `ENGINE=ROCKSDB`.

(TokuDB, Percona's earlier alternative engine, is still present at `storage/tokudb/` but its
build is hard-disabled — `CMakeLists.txt` opens with `RETURN()`. Engines come and go; the
handler API abides.)

## 11.3 Security & operations plugins/components

The plugin API (same mechanism that registers engines, Ch. 6) carries most of the rest:

| feature | location | one-liner |
|---------|----------|-----------|
| Audit log | `plugin/audit_log/`, `plugin/audit_log_filter/` | file/syslog audit trail of connections & statements |
| PAM authentication | `plugin/percona-pam-for-mysql/` | OS-level auth (LDAP/2FA via PAM stack) |
| Keyring for Vault | `plugin/keyring_vault/`, `components/keyrings/` | encryption keys in HashiCorp Vault (TDE key management) |
| Data masking | `components/masking_functions/` | masking/anonymization UDFs |
| PROCFS | `plugin/procfs/` | host `/proc` metrics via I_S — observability without shell access |
| Backup locks | MDL `BACKUP_TABLES`/`BACKUP_LOCK` namespaces (Ch. 7) | `LOCK TABLES FOR BACKUP` — the light-weight FTWRL replacement XtraBackup uses |
| Column compression | DD system views `compression_dictionary*` (Ch. 10) | per-column compression with shared dictionaries |

## 11.4 Observability patches

The oldest and most-loved Percona deltas are diagnostic:

- **Extended slow log** (`sql/log.cc`, `log_slow_verbosity`): per-query InnoDB stats — pages
  read, I/O wait, lock wait — in the slow log; pre-dates performance_schema and still often
  more convenient. `pt-query-digest` grew up on this format.
- **User/table/index statistics** (`sql/userstat.cc`): I_S tables (`TABLE_STATISTICS`,
  `INDEX_STATISTICS`, `USER_STATISTICS`) counting rows read/changed per object — the quick
  answer to "which index is dead weight?"

## 11.5 What to remember

1. Percona's delta maps onto the architecture's four sanctioned extension points:
   connection handler, storage engine, plugin/component APIs, and instrumentation. Almost
   nothing forks the core query path — that's what keeps the fork maintainable.
2. MyRocks vs InnoDB is the cleanest real-world case study of the handler abstraction: LSM
   vs B+tree behind one interface, with memcomparable key encoding as the bridge.
3. The thread pool solves the C10K problem Chapter 1's model can't; backup locks solve the
   FTWRL problem Chapter 7 explained.
4. When evaluating "Percona vs upstream," you're mostly evaluating these tables — the SQL
   core is the same code you studied in Chapters 1-10.

**Try it:** `SHOW PLUGINS` on a Percona server, then match each row to a directory in
`plugin/` or `storage/`; `SET GLOBAL log_slow_verbosity='full'` and read one slow-log entry.

---
**Previous:** [Chapter 10 — The Data Dictionary](./10-data-dictionary.md) · **Next:** [Chapter 12 — Capstone](./12-journey-capstone.md)
