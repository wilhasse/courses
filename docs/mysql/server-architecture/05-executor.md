# Chapter 5 — The Iterator Executor

> How the plan actually runs: the volcano model, MySQL-style — plus filesort and internal
> temporary tables.
> Source: `sql/sql_union.cc`, `sql/iterators/`, `sql/join_optimizer/access_path.cc`,
> `sql/filesort.cc`, `sql/sql_tmp_table.cc`

## 5.1 The execution loop is ten lines

Since 8.0.20+ MySQL executes every query the same way
(`Query_expression::ExecuteIteratorQuery`, `sql/sql_union.cc:1682`):

```c
m_root_iterator->Init();                       // :1779
for (;;) {
    int error = m_root_iterator->Read();       // 0 = row, -1 = EOF, 1 = error
    if (error) break;
    query_result->send_data(thd, *fields);     // row is in table->record[0] / Items
}
query_result->send_eof(thd);
```

Everything a query does — joins, aggregation, sorting, subqueries — is inside that
`Read()` call, in a tree of **iterators**. This replaced the legacy pre-8.0.20 executor
(a fixed nested-loop machine with special cases braided through it) with the textbook
Volcano model: each operator pulls rows from its children.

## 5.2 The RowIterator contract and the zoo

`RowIterator` (`sql/iterators/row_iterator.h:82`) is a minimal interface: `Init()`
(may be called again to rewind — that's how the inner side of a nested loop restarts),
`Read()` (produce one row into the table record buffers), plus `SetNullRowFlag` (outer-join
NULL rows) and `UnlockRow` (release a lock on a row that failed the WHERE — the MVCC/locking
interplay from the InnoDB series surfacing here).

Iterators come in three families (`sql/iterators/`):

| family | examples | role |
|--------|----------|------|
| leaf | `TableScanIterator` (`basic_row_iterators.h:56`), `IndexScanIterator`, `RefIterator`, `EQRefIterator`, `IndexRangeScanIterator`, `ConstIterator` | fetch rows via the handler API (Ch. 6) |
| join | `NestedLoopIterator` (`composite_iterators.h:323`), `HashJoinIterator` (`hash_join_iterator.h:253`), `BKAIterator` | combine two children |
| shape | `FilterIterator`, `SortingIterator`, `AggregateIterator`, `LimitOffsetIterator`, `MaterializeIterator` (`composite_iterators.cc:621`), `StreamingIterator`, `WeedoutIterator`, `WindowIterator` | transform a stream |

`CreateIteratorFromAccessPath()` (`sql/join_optimizer/access_path.cc:378`) maps the
optimizer's AccessPath tree 1:1 onto iterators — a giant switch, deliberately iterative
rather than recursive to bound stack depth. `EXPLAIN FORMAT=TREE` output *is* this tree;
`EXPLAIN ANALYZE` wraps every node in a `TimingIterator` and prints actual rows/timings.

A `SELECT ... JOIN ... WHERE ... ORDER BY` becomes, literally:

```
SortingIterator
└── NestedLoopIterator
    ├── FilterIterator( t1.x > 10 )
    │   └── TableScanIterator(t1)
    └── RefIterator(t2, key=t1.id)     ← Init() re-run per t1 row
```

**Hash join** (8.0.18+) deserves a note: `HashJoinIterator` builds an in-memory hash table
from one side (spilling to disk chunks when `join_buffer_size` is exceeded) and probes with
the other — finally giving MySQL a join that doesn't need an index, and quietly replacing
the old "block nested loop" buffering.

## 5.3 filesort: ORDER BY without an index

When no index provides the order, `SortingIterator` wraps **filesort**
(`filesort(...)`, `sql/filesort.cc:367`) — a classic external sort:

1. Pull all rows from the child iterator, building fixed-format **sort keys**
   (`make_sortkey`) in a buffer of `sort_buffer_size`.
2. Buffer full → sort it, write a run (`Merge_chunk`) to a temp file; repeat.
3. Merge runs (`merge_many_buff` → `merge_buffers`, `:1918`) down to one sorted stream.

The payload question — what travels with the key — has two answers
(`Addon_fields_status`, `sql/sort_param.h:49`): **addon fields** (pack the needed columns
with the key; no second table visit) or **row IDs** (sort small entries, then re-fetch each
row — required when rows are too wide or contain big BLOBs). That choice is the difference
between one and two passes over the data, and you can observe it in
`optimizer_trace`'s `filesort_summary`.

## 5.4 Internal temporary tables

`MaterializeIterator`, `TemptableAggregateIterator`, window buffering, UNION/CTE
materialization — all need scratch tables (`create_tmp_table`, `sql/sql_tmp_table.cc:886`).
The engine ladder (`setup_tmp_table_handler`, `:2102`):

```
in-memory: TempTable engine (storage/temptable/, default; supports VARCHAR/BLOB)
              — or legacy MEMORY engine (internal_tmp_mem_storage_engine)
   │ exceeds temptable_max_ram / row constraints  (use_tmp_disk_storage_engine :2044)
   ▼
on-disk: InnoDB temp tablespace (ibtmp1)
```

"Using temporary; Using filesort" in EXPLAIN — the two most famous words in MySQL
performance work — refer precisely to sections 5.3 and 5.4.

## 5.5 What to remember

1. One execution model: AccessPath → iterator tree → `Init()`/`Read()` pull loop. Rows
   materialize in `table->record[0]` buffers; `Query_result` streams them to the client
   as produced (Ch. 1).
2. The iterator names in `EXPLAIN FORMAT=TREE` are literal C++ classes in
   `sql/iterators/` — the plan you read is the code that runs.
3. filesort = external merge sort; addon-fields vs row-id decides whether sorted rows must
   be re-fetched. Hash join covers index-less equijoins with disk spill.
4. Temp tables cascade TempTable → InnoDB-on-disk; most "why is this query slow" answers
   live in 5.3/5.4.

**Try it:** `EXPLAIN ANALYZE SELECT ... ORDER BY ...` — match every line to an iterator
class, then find its `Read()` in `sql/iterators/` and read it. Most are under a page.

---
**Previous:** [Chapter 4 — The Optimizer](./04-optimizer.md) · **Next:** [Chapter 6 — The Handler API](./06-handler-api.md)
