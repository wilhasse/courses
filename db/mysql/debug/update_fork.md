# Git

From time to time is good to keep my fork updated

First make sure you have upstream

```bash
@mysql-8-src:/data/percona-server$ git remote -v
origin  https://wilhasse/wilhasse/percona-server (fetch)
origin  https://wilhasse/wilhasse/percona-server (push)

@mysql-8-src:/data/percona-server$ git remote add upstream https://github.com/percona/percona-server
```

Merge commits from upstream

```bash
@mysql-8-src:/data/percona-server$ git fetch upstream
git merge upstream/8.0
remote: Enumerating objects: 31234, done.
remote: Counting objects: 100% (15658/15658), done.
remote: Compressing objects: 100% (2658/2658), done.
Receiving objects: 100% (31234/31234), 193.61 MiB | 29.20 MiB/s, done.
..
```

Push to git

```bash
cslog@mysql-8-src:/data/percona-server$ git push
Enumerating objects: 4, done.
Counting objects: 100% (4/4), done.
Delta compression using up to 8 threads
Compressing objects: 100% (2/2), done.
Writing objects: 100% (2/2), 336 bytes | 336.00 KiB/s, done.
Total 2 (delta 1), reused 0 (delta 0), pack-reused 0
remote: Resolving deltas: 100% (1/1), completed with 1 local object.
To https://github.com/wilhasse/percona-server
   d7fc4f533fd..66546094354  8.0 -> 8.0
```
