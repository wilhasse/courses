# Chapter 3 — Resolution & Prepare

> Binding names to tables and columns, opening tables, and the permanent query
> transformations — the phase between parsing and optimization.
> Source: `sql/sql_resolver.cc`, `sql/sql_base.cc`, `sql/table.h`, `sql/sql_prepare.cc`

## 3.1 Where we are

The parser produced an `Sql_cmd` (Chapter 2). For DML, `Sql_cmd_dml::prepare()`
(`sql/sql_select.cc:495`) now runs — once per statement, or once *ever* for a prepared
statement:

```
Sql_cmd_dml::prepare
 ├─ open_tables_for_query()        (sql/sql_base.cc:6829)   tables + MDL
 └─ prepare_inner → Query_block::prepare()  (sql/sql_resolver.cc:180)
      name resolution + permanent transformations
```

## 3.2 Opening tables: shares, instances, references

Three layers of "table" (all `sql/table.h`) — confuse them and no server code makes sense:

| struct | one per | contents |
|--------|---------|----------|
| `TABLE_SHARE` (`table.h:692`) | table *definition*, process-wide, cached & refcounted | columns, keys, version — loaded from the data dictionary (Ch. 10) |
| `TABLE` (`table.h:1431`) | *open instance*, per (session × table use) | the `handler* file` (Ch. 6), record buffers, read/write bitmaps |
| `Table_ref` (`table.h:2872`) | *mention in the query* (FROM entry, formerly TABLE_LIST) | name, alias, links into the query block, `mdl_request` |

`open_tables_for_query()` walks the statement's `Table_ref` list: acquire the **metadata
lock** (`Table_ref::mdl_request`, `table.h:3949` — DML takes `MDL_SHARED_READ/WRITE`; the MDL
system is Chapter 7), find/load the `TABLE_SHARE`, get a `TABLE` from the per-session table
cache, and point `Table_ref::table` at it. Views and derived tables expand here too.

## 3.3 `Query_block::prepare()`: name resolution

`sql/sql_resolver.cc:180` runs a fixed sequence per query block (recursing into subqueries):

1. `setup_tables()` (`:1175`) — assign table numbers/maps, build the leaf-table list.
2. `resolve_placeholder_tables()` (`:1300`) — resolve derived tables and views; **merge**
   mergeable ones into the parent block (`merge_derived`, `:3473`) so
   `SELECT * FROM (SELECT ...)` can often optimize as a plain join.
3. `setup_wild()` (`:1629`) — expand `*` into concrete `Item_field`s.
4. `setup_fields()` (`sql/sql_base.cc:9063`) — run **`fix_fields()`** on the select list:
   every `Item_field` finds its `Field` in an opened `TABLE`; types, nullability, collations
   are derived bottom-up. Ambiguous or unknown columns error here.
5. `setup_conds()` (`:1685`) — same for join conditions and WHERE.
6. `setup_group()` / HAVING / `setup_order()` / `resolve_limits` — with SQL's peculiar
   scoping (HAVING sees aliases, ORDER BY sees both...) encoded in
   `Name_resolution_context`.

## 3.4 Permanent transformations

Interleaved with resolution, the server performs rewrites that are **structural and
irreversible** — done once, valid for every future execution (which is precisely why they
live here and not in the optimizer):

- **Outer-join simplification** (`simplify_joins`, `:1935`): `LEFT JOIN` becomes inner join
  when the WHERE clause rejects NULL-complemented rows — the classic
  `... LEFT JOIN t2 ... WHERE t2.x = 5` case.
- **Subquery → semijoin** (`resolve_subquery` `:1384`, `flatten_subqueries` `:3822`,
  `convert_subquery_to_semijoin` `:3009`): `WHERE id IN (SELECT ...)` is pulled up into the
  outer block as a *semijoin nest*, letting the join optimizer choose among execution
  strategies (Chapter 4) instead of executing the subquery per row.
- **Scalar subquery → derived table** (`transform_scalar_subqueries_to_join_with_derived`,
  `:7681`) and IN→EXISTS injection for subqueries that can't be flattened.
- **Redundant clause removal** in subqueries (`remove_redundant_subquery_clauses`, `:4151`) —
  an ORDER BY inside an IN-subquery means nothing, so it's deleted.

After `prepare()`, the `Query_block` is fully bound and normalized. `Sql_cmd_dml::execute()`
(`sql/sql_select.cc:663`) can now run it — going next to the optimizer.

## 3.5 Prepared statements: why this phase is separable

The prepare/execute split is not just an API nicety — it is *this* boundary:

- `Prepared_statement` (`sql/sql_prepare.h:346`) parses + resolves **once**, keeping the LEX
  and Item trees alive in their own `MEM_ROOT` (`m_arena`); `?` placeholders are
  `Item_param`s collected in `lex->param_list`.
- `execute_loop()` (`sql/sql_prepare.cc:2993`) binds parameter values and reruns
  optimization + execution against the saved trees.
- The catch: cached resolution can go **stale** (table altered, index dropped). Every open
  validates the `TABLE_SHARE` version; a mismatch raises `ER_NEED_REPREPARE`, caught by the
  `Reprepare_observer`, and `Prepared_statement::reprepare()` (`:3216`) transparently
  re-parses, re-resolves, swaps in the new tree, and retries — invisible to the client.

This is also why the permanent-vs-per-execution split matters: transformations in this
chapter survive across executions; everything the optimizer does (next chapter) is redone
per execution.

## 3.6 What to remember

1. Three "table" structs: `TABLE_SHARE` (cached definition), `TABLE` (open instance with the
   engine `handler`), `Table_ref` (query mention holding the MDL request).
2. `fix_fields()` is where names become columns and types are derived — the moment SQL text
   semantics is finally pinned down.
3. Big rewrites (outer-join reduction, subquery flattening/semijoins, derived-table merging)
   happen here, permanently — the optimizer never sees the original shape.
4. Prepared statements = "freeze the output of this chapter, redo everything after it",
   with automatic re-prepare on metadata changes.

**Try it:** `EXPLAIN` a `WHERE x IN (SELECT ...)` query — the plan shows the semijoin
(`FirstMatch`/`MaterializeLookup`...), proof the subquery ceased to exist in this phase.

---
**Previous:** [Chapter 2 — The Parser](./02-parser.md) · **Next:** [Chapter 4 — The Optimizer](./04-optimizer.md)
