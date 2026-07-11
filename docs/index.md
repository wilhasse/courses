# wilhasse's Documentation Hub

Deep-dive courses, project documentation, and study notes — all in one place.
Each collection below is a self-contained folder in the
[courses repo](https://github.com/wilhasse/courses); push markdown, and it appears here.

## The Query Optimization Journey

<div class="grid cards" markdown>

- **[Making complex analytical queries fast on MySQL](query-optimization/index.md)**

    The multi-year thread that ties this whole site together: the idea and its
    constraints, everything studied, the five strategies and labs (TiDB, Doris, InnoDB
    parsing, PolarDB-X…), and the two systems that emerged — **SmartSQL** (in
    production) and **cslog-query** (in development).

</div>

## MySQL Internals

Two step-by-step courses covering a complete MySQL database — from the bytes on disk to
the replica stream — grounded in real source code with `file.c:line` references and
diagrams throughout. Written to consolidate the journey's studies.

<div class="grid cards" markdown>

- **[MySQL Server Architecture Deep-Dive](mysql/server-architecture/README.md)**

    *Everything above the engine, top-down.* Based on Percona Server 8.0: connections,
    parser, optimizer, iterator executor, handler API, two-phase commit, binlog,
    replication, and the 8.0 data dictionary — 13 chapters.

    📄 [PDF version](mysql/server-architecture/mysql-server-architecture.pdf)

- **[InnoDB Architecture Deep-Dive](mysql/innodb-architecture/README.md)**

    *The storage engine, bottom-up.* Based on Embedded InnoDB 1.0.6 (the pre-Oracle
    Innobase codebase): tablespaces, pages, buffer pool, redo log & recovery, B+trees,
    MVCC, locking, and background threads — 13 chapters.

    📄 [PDF version](mysql/innodb-architecture/innodb-architecture.pdf)

</div>

**How to read them:** new to database internals? Start with the InnoDB series — it builds
a complete storage engine in your head from first principles. Know storage engines and
want the SQL layer? Start with the server series. The two meet at the **handler API**
(server series, Chapter 6), and the final
[capstone](mysql/server-architecture/12-journey-capstone.md) traces one UPDATE through
every layer of both.

## Notes

- [Study notes & yearly summaries](notes/index.md) — course summaries and shorter
  write-ups.

## Adding a new collection

1. Create a folder under `docs/` (e.g. `docs/my-project/`) with an `index.md` or
   `README.md` plus your pages.
2. Add a section to `nav:` in `mkdocs.yml` and a card/link on this page.
3. Push to `main` — GitHub Actions rebuilds and deploys the site automatically.
