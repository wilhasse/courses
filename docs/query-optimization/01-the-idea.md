# The Idea

> The problem that started everything, and the constraints that shaped every attempt.

## The problem

A production system runs on **Percona Server (MySQL) + InnoDB**. The OLTP side is fine —
InnoDB is excellent at transactional work. The pain is **complex analytical queries**:
reports joining 10+ tables over large datasets, running on MySQL read replicas, taking
minutes to hours. The worst offenders run for *hours* on hardware that can't simply be
upgraded away — these reports run at customer sites, on modest machines.

Why MySQL struggles here is not a mystery (the
[deep-dive courses](../mysql/server-architecture/README.md) on this site exist because of
this problem):

- The optimizer's cost model and greedy join-order search degrade with many tables —
  a bad join order on a 10-table query can cost orders of magnitude.
- Fixing it with indexes requires composite indexes per query shape — a combinatorial
  explosion nobody can maintain.
- InnoDB is a **row store**: analytical scans drag entire rows through the buffer pool
  to read three columns.

## The constraints (what made this hard)

If it were just "use an OLAP database," there'd be no journey. The real constraints:

1. **The data must stay in MySQL.** The OLTP application is mature and stays. Anything
   analytical must feed off MySQL, not replace it.
2. **The FK-cascade trap.** The schema uses foreign keys with cascading deletes — and
   **cascaded changes don't appear in the binlog**. Any external system replicating via
   CDC silently diverges. This single detail disqualified naive "replicate to X" setups
   and haunted every strategy.
3. **Operational simplicity.** Solutions run at customer sites: no clusters to babysit,
   no fragile sync pipelines (a CDC pipeline that breaks weekly is worse than a slow
   report).
4. **Application transparency.** Ideally, clients keep speaking the MySQL protocol to
   something that looks exactly like MySQL.

## The thesis that emerged

After every study and lab, the same shape kept winning:

> **Keep MySQL as the source of truth. Put a columnar/analytical engine next to it —
> fed by the binlog, hidden behind the MySQL wire protocol — and route only the hard
> queries to it.**

Everything else in this collection is the story of earning that sentence: the
[studies](./02-studies.md) that built the necessary understanding, the
[labs and dead ends](./03-labs.md) that eliminated alternatives, and the two systems
that implement it — [SmartSQL](./04-smart-sql.md) (accelerate queries against MySQL)
and [cslog-query](./05-cslog-query.md) (move reports off MySQL).

---
**Back to:** [The Journey](./index.md) · **Next:** [What I Studied](./02-studies.md)
