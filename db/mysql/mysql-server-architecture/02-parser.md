# Chapter 2 — The Parser

> SQL text → tokens → parse tree → LEX. The first transformation of a query's life.
> Source: `sql/sql_lex.cc`, `sql/sql_yacc.yy`, `sql/parse_tree_nodes.h/.cc`, `sql/sql_cmd.h`

## 2.1 Three representations, two steps

MySQL's front end turns a query string into three successive data structures:

```
"SELECT a FROM t WHERE b>1"
   │  lexer: Lex_input_stream + MYSQLlex()          (sql/sql_lex.cc:1310)
   ▼
tokens (SELECT_SYM, IDENT, FROM, IDENT, WHERE, IDENT, '>', NUM)
   │  Bison grammar: sql/sql_yacc.yy (~18,500 lines)
   ▼
Parse Tree: PT_select_stmt → PT_query_expression → PT_query_specification → ...
   │  contextualize(): PT_select_stmt::make_cmd()  (sql/parse_tree_nodes.cc:707)
   ▼
LEX + Query_expression / Query_block + Item trees      ← what the rest of the server uses
   └─ plus an Sql_cmd object (e.g. Sql_cmd_select) that will drive prepare/execute
```

## 2.2 The lexer

`Lex_input_stream` (`sql/sql_lex.h:3227`) is a hand-written scanner: a cursor over the query
buffer (`yyGet`/`yyPeek`/`yyUnget` primitives), a state machine in `lex_one_token()`
(`sql/sql_lex.cc:1412`), keyword lookup, and two notable extras:

- **Digest computation on the fly** (`add_digest_token`, `sql/sql_lex.cc:386`): as tokens
  stream by, a normalized statement digest is built — this is where
  `performance_schema.events_statements_summary_by_digest` gets its query fingerprints.
  The parser and the observability system share one pass.
- **Token lookahead** for grammar warts: `MYSQLlex()` (`sql_lex.cc:1310`) buffers one token to
  disambiguate cases like `WITH ROLLUP` (`:1383-1407`) — a reminder that SQL was not designed
  for LALR(1) parsers.

## 2.3 The grammar and the parse tree

`sql/sql_yacc.yy` is one of the largest Bison grammars in production use. Its actions do
**no semantic work** — each rule just allocates a `PT_*` node on the parse arena
(`NEW_PTN`). The SELECT chain, top-down:

```
select_stmt (:9933)          → PT_select_stmt
 └─ query_expression (:10029)  → PT_query_expression    (+ WITH clause = CTEs)
     └─ query_expression_body (:10045)  UNION/EXCEPT/INTERSECT nesting
         └─ query_primary (:10085)
             └─ query_specification (:10106) → PT_query_specification
                  SELECT options, item list, FROM, WHERE, GROUP BY, HAVING, windows
```

**Why the intermediate tree?** Older MySQL built `LEX` structures directly in grammar actions,
which made the grammar unmaintainable (actions ran in grammar-reduction order, not semantic
order) and blocked reuse. 8.0's two-phase design — *build a dumb tree, then walk it* — is the
textbook approach, arrived at 20 years late and still being migrated command by command
(look at how many `SQLCOM_*` cases in `mysql_execute_command` still don't go through
`Sql_cmd`).

The walk is called **contextualization**: `PT_select_stmt::make_cmd()`
(`sql/parse_tree_nodes.cc:707`) creates a `Parse_context` and calls `contextualize()`
recursively. Each node populates the semantic structures — e.g.
`PT_query_specification::contextualize()` (`:1170`) fills the current `Query_block`'s select
list, table list, WHERE and HAVING. The output is an `Sql_cmd_select` stored in
`lex->m_sql_cmd`, whose `prepare()`/`execute()` (Chapter 3) drive everything else.

## 2.4 The output structures: LEX, Query_block, Item

Three families you must know to read any `sql/` code:

- **`LEX`** (`sql/sql_lex.h:3753`) — the per-statement descriptor: `sql_command`
  (the `SQLCOM_*` verb), the tree of query blocks, the parameter list, flags. One per
  statement, hung off `thd->lex`.
- **`Query_expression` / `Query_block`** (`sql/sql_lex.h:623`, `:1160`) — the query's shape.
  A *block* is one SELECT (fields, table list, where/having, group/order); an *expression*
  is one or more blocks combined by UNION/EXCEPT/INTERSECT plus ORDER/LIMIT. Subqueries nest:
  an inner `Query_expression` hangs under an `Item_subselect` in the outer block. Every later
  phase — resolver, optimizer, executor — is organized around walking this tree.
- **`Item`** (`sql/item.h:853`) — every expression node: `Item_field` (a column ref),
  `Item_func_*` (functions/operators), `Item_cond_and/or`, literals, `Item_param` (a `?`
  placeholder), `Item_subselect`. Items are created "unresolved" by the parser; the key
  virtual `fix_fields()` (`item.h:1180`) — run in the next chapter — binds names to real
  columns and derives types. An item can *replace itself* through the `Item**` reference
  parameter, which is how many rewrites are implemented.

All of it is allocated on the statement `MEM_ROOT` — no destructors, freed wholesale when the
statement ends (or kept, for prepared statements — Chapter 3).

## 2.5 What to remember

1. Parsing is two phases: a Bison grammar building a dumb `PT_*` tree, then
   `contextualize()` producing the semantic `LEX`/`Query_block`/`Item` structures and an
   `Sql_cmd`. The split exists because the one-phase design collapsed under its own weight.
2. `Query_block` (one SELECT) and `Query_expression` (blocks + set operations) are the
   backbone data structure of the entire SQL layer — learn their fields once, use them in
   every chapter after.
3. `Item` trees are the universal expression currency, unresolved until `fix_fields()`.
4. Statement digests (your `performance_schema` fingerprints) are computed inside the lexer.

**Try it:** `EXPLAIN FORMAT=JSON` names query blocks (`query_block`, `nested_loop`…) exactly as
they exist in this tree; and `SELECT STATEMENT_DIGEST_TEXT('SELECT a FROM t WHERE b>1');`
shows the lexer's normalization directly.

---
**Previous:** [Chapter 1 — Connections & Dispatch](./01-connections-and-dispatch.md) · **Next:** [Chapter 3 — Resolution & Prepare](./03-resolver-prepare.md)
