# Source

Percona fork  
https://github.com/k1n9met/percona-server/tree/8.0.36-28-pq  

Huawei  
https://gitee.com/kunpengcompute/mysql-server/tree/KunpengBoostKit22.0.RC2.ParallelQuery

# Merge

Clone

```bash
# Parallel Query fork
git clone -b 8.0.36-28-pq https://github.com/k1n9met/percona-server.git percona-server-pq

# Original Percona 8.0.39
git clone -b release-8.0.39 https://github.com/percona/percona-server.git percona-8.0.39

# Add 8.0.36 Percona to pq fork
cd percona-server-pq/
git remote add upstream https://github.com/percona/percona-server.git
git fetch upstream
git branch
```

Generate patch

```bash
# From your fork's directory
git diff upstream/release-8.0.36-28 > changes.diff
ls -la changes.diff 
cd ..
```

Apply

```bash
cd percona-server-8.0.39/
git apply --check ../percona-server-pq/changes.diff
patch p1 < ../percona-server-pq/changes.diff
git apply --reject ../percona-server-pq/changes.diff
```

Check rejects, apply manually

```bash
# Example
/home/cslog/percona-server-8.0.39/p1.rej
/home/cslog/percona-server-8.0.39/sql/sql_class.h.rej
/home/cslog/percona-server-8.0.39/sql/sql_lex.h.rej
/home/cslog/percona-server-8.0.39/sql/sql_select.cc.rej
/home/cslog/percona-server-8.0.39/storage/innobase/row/row0pread.cc.rej
```