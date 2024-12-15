<?php

namespace App\Livewire\Auth;

use App\Providers\RouteServiceProvider;
use Illuminate\Support\Facades\Auth;
use Livewire\Component;

class Login extends Component
{
    public $login = '';
    public $password = '';
    public $remember = false;

    protected $rules = [
        'login' => 'required',
        'password' => 'required',
    ];

    public function mount()
    {
        logger('Component mounted');
    }

    public function updated($property)
    {
        logger('Property updated: ' . $property);
    }

    public function authenticate()
    {
        logger('Login method called');
        logger('Login data', ['login' => $this->login, 'password' => strlen($this->password)]);
        
        try {
            $this->validate();

            if (Auth::attempt(['username' => $this->login, 'password' => $this->password], $this->remember)) {
                session()->regenerate();
                return redirect()->intended(RouteServiceProvider::HOME);
            }

            $this->addError('login', 'Invalid credentials');
            logger('Authentication failed');
            
        } catch (\Exception $e) {
            logger('Login error', ['error' => $e->getMessage()]);
            $this->addError('login', 'An error occurred during login');
        }
    }

    public function render()
    {
        return view('livewire.auth.login')
            ->layout('components.layouts.app');
    }
} 