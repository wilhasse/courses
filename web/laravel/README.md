# Base

PHP
https://windows.php.net/download  

Composer
https://getcomposer.org/download/  

Node.js
https://nodejs.org/en/download/package-manager

# Install Laravel

php.ini, enable:

```ini
extension=fileinfo
extension=zip
```

Create a new project and install livewire:

```bash
composer create-project laravel/laravel newapp
composer require livewire/livewire
```

If already exist and you forked the project
install modules and create a new env file

```bash
D:\courses\web\laravel\newapp>composer install
D:\courses\web\laravel\newapp>copy .env.example .env
        1 arquivo(s) copiado(s).

D:\courses\web\laravel\newapp>php artisan key:generate

   INFO  Application key set successfully.
```

# Start server

Start new server

```bash
C:\courses\web\laravel\newapp>php artisan serve
  forking is not supported on this platform

   INFO  Server running on [http://127.0.0.1:8000].

  Press Ctrl+C to stop the server

  2024-12-15 08:51:52 / ........................................................................................ ~ 1s
```

Vite integration to hot reload

```bash
C:\courses\web\laravel\newapp>npm install
C:\courses\web\laravel\newapp>npm run dev
> dev
> vite


  VITE v6.0.3  ready in 209 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help

  LARAVEL v11.35.1  plugin v1.1.1
```


You will get an internal server error

Illuminate\Database\QueryException
could not find driver (Connection: sqlite, SQL: PRAGMA foreign_keys = ON;)

# MySQL Config


```ini
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=your_database_name
DB_USERNAME=your_mysql_username
DB_PASSWORD=your_mysql_password
```

php artisan config:clear

php ini uncomment:

```ini
extension=pdo_mysql
extension=mysqli
extension=intl
```

Verify restart server

```bash
php -m
# stop server ctrl+c
php artisan serve
```

Migrate data

```bash
D:\courses\web\laravel\newapp>php artisan migrate

   INFO  Preparing database.

  Creating migration table .............................................................................. 56.74ms DONE

   INFO  Running migrations.

  0001_01_01_000000_create_users_table ................................................................. 125.22ms DONE
  0001_01_01_000001_create_cache_table .................................................................. 33.93ms DONE
  0001_01_01_000002_create_jobs_table ................................................................... 71.85ms DONE


D:\courses\web\laravel\newapp>php artisan db:show

  MySQL .................................................................................................... 8.0.36-28
  Connection ................................................................................................... mysql
  Database ................................................................................................... laravel
  Host ..................................................................................................... 10.1.0.10
  Port .......................................................................................................... 3306
  Username ...................................................................................................... root
  URL ................................................................................................................
  Open Connections ................................................................................................. 1
  Tables ........................................................................................................... 9
```

Generate php create tables for migration from already existed tables

```bash
# install tool
composer require --dev kitloong/laravel-migrations-generator

# only specific tables
php artisan migrate:generate --tables=customer,date,lineorder,part,supplier

Setting up Tables and Index migrations.
Created: D:\courses\web\laravel\newapp\database/migrations/2024_12_16_013644_create_customer_table.php
Created: D:\courses\web\laravel\newapp\database/migrations/2024_12_16_013644_create_date_table.php
Created: D:\courses\web\laravel\newapp\database/migrations/2024_12_16_013644_create_lineorder_table.php
Created: D:\courses\web\laravel\newapp\database/migrations/2024_12_16_013644_create_part_table.php
Created: D:\courses\web\laravel\newapp\database/migrations/2024_12_16_013644_create_supplier_table.php
```

# Debug

XDebug  
https://xdebug.org/download

save to c:\php-8.4.1\ext\php_xdebug.dll

php.ini add:

```ini
[xdebug]
zend_extension=xdebug
xdebug.mode=debug
xdebug.start_with_request=yes
xdebug.client_port=9003
xdebug.client_host=127.0.0.1
```

test

```bash
D:\courses\web\laravel\newapp>php -v
PHP 8.4.1 (cli) (built: Nov 20 2024 11:13:22) (NTS Visual C++ 2022 x64)
Copyright (c) The PHP Group
Zend Engine v4.4.1, Copyright (c) Zend Technologies
    with Xdebug v3.4.0, Copyright (c) 2002-2024, by Derick Rethans
```

json in vscode

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Listen for Xdebug",
            "type": "php",
            "request": "launch",
            "port": 9003,
            "pathMappings": {
                "/path/to/your/project": "${workspaceFolder}"
            }
        }
    ]
}
```