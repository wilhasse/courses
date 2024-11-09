# Intro

PostgreSQL run on GPUs  
https://heterodb.github.io/pg-strom/

# Install

Guide  
https://heterodb.github.io/pg-strom/install/

Nvidia

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
apt-get install cuda
reboot
```

PostgreSQL

```bash
apt-get install postgresql
mkdir /var/lib/pgdata
chown postges:postgres /var/lib/pgdata
chown postgres:postgres /var/lib/pgdata
su - postgres

/usr/pgsql-16/bin/initdb -D /var/lib/pgdata/
```

Pg-strom

```bash
git clone https://github.com/heterodb/pg-strom.git
cd pg-strom/src
make PG_CONFIG=/usr/pgsql-16/bin/pg_config
sudo make install PG_CONFIG=/usr/pgsql-16/bin/pg_config

wget https://heterodb.github.io/swdc/deb/heterodb-extra_5.4-1_amd64.deb
dpkg -i heterodb-extra_5.4-1_amd64.deb
rm heterodb-extra_5.4-1_amd64.deb
```

# Config

Edited /etc/postgresql/16/main/postgresql.conf

```conf
#added
shared_preload_libraries = '$libdir/pg_strom'
max_worker_processes = 100
shared_buffers = 10GB
work_mem = 1GB
```

Log /var/log/postgresql/postgresql-16-main.log

```plain 
2024-09-20 22:27:46.069 -03 [7671] LOG:  PG-Strom binary built for CUDA 12.6 (CUDA runtime 12.6)
2024-09-20 22:27:46.069 -03 [7671] LOG:  PG-Strom: GPU0 NVIDIA GeForce RTX 4090 (128 SMs; 2520MHz, L2 73728kB), RAM 23.55GB (384bits, 10.01GHz), PCI-E Bar1 32GB, CC 8.9
2024-09-20 22:27:46.079 -03 [7671] LOG:  starting PostgreSQL 16.4 (Ubuntu 16.4-0ubuntu0.24.04.2) on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 13.2.0-23ubunt
```

Extension

```psql
teste=# \dx
                Lista de extensões instaladas
  Nome   | Versão |  Esquema   |          Descrição
---------+--------+------------+------------------------------
 plpgsql | 1.0    | pg_catalog | PL/pgSQL procedural language
(1 linha)

teste=# CREATE EXTENSION pg_strom;
CREATE EXTENSION
teste=# \dx
                                  Lista de extensões instaladas
   Nome   | Versão |  Esquema   |                           Descrição
----------+--------+------------+----------------------------------------------------------------
 pg_strom | 5.1    | public     | PG-Strom - big-data processing acceleration using GPU and NVME
 plpgsql  | 1.0    | pg_catalog | PL/pgSQL procedural language
(2 linhas)

postgres=# select gpu_id,att_Name,att_value from pgstrom.gpu_device_info;
 gpu_id |                   att_name                   |                att_value
--------+----------------------------------------------+------------------------------------------
      0 | DEV_NAME                                     | NVIDIA GeForce RTX 4090
      0 | DEV_ID                                       | 0
      0 | DEV_UUID                                     | GPU-1080144e-dc83-ae03-273e-5fb04fdd1f0f
      0 | DEV_TOTAL_MEMSZ                              | 23.55GB
      0 | DEV_BAR1_MEMSZ                               | 32.00GB

teste=#
SET pg_strom.enabled = on;
SET pg_strom.gpu_setup_cost = 0;
SET pg_strom.gpu_operator_cost = 0.01;
SET pg_strom.cpu_fallback = off;
SET max_parallel_workers_per_gather = 0;

```

# Error

Log: /var/log/postgresql/postgresql-16-main.log

```plain
2024-09-21 09:06:33.840 -03 [24022] ERROR:  failed on the build process at [/tmp/.pgstrom_fatbin_build_TpicZ9]
2024-09-21 09:06:33.842 -03 [14751] LOG:  background worker "PG-Strom GPU Service" (PID 24022) exited with exit code 1
```

Check PATH environment in PostgreSQL

plpython3

```bash
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
apt-get update
apt-get install postgresql-plpython3-16
```

```psql
CREATE EXTENSION IF NOT EXISTS plpythonu;
CREATE OR REPLACE FUNCTION show_path() RETURNS text AS $$
    import os
    return os.environ['PATH']
$$ LANGUAGE plpythonu;

SELECT show_path();
```

Pass PATH environment to PostgreSQL

```bash
#add in /etc/postgresql/16/main/environment
PATH='/usr/local/bin:/usr/bin:/bin'

systemctl daemon-reload
systemctl restart postgresql@16-main.service
```

