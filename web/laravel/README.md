# Install

PHP
https://windows.php.net/download  

Composer
https://getcomposer.org/download/  

# Laravel

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

Start new server

```bash
C:\courses\web\laravel\newapp>php artisan serve
  forking is not supported on this platform

   INFO  Server running on [http://127.0.0.1:8000].

  Press Ctrl+C to stop the server

  2024-12-15 08:51:52 / ........................................................................................ ~ 1s
```

You will get an internal server error

Illuminate\Database\QueryException
could not find driver (Connection: sqlite, SQL: PRAGMA foreign_keys = ON;)
