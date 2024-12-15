<?php

use Illuminate\Support\Facades\Route;
use App\Livewire\Auth\Login;

Route::get('/', function () {
    return view('welcome');
});

Route::middleware('guest')->group(function () {
    Route::get('login', Login::class)->name('login');
});
