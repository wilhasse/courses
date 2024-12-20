<?php

namespace App\Livewire\Auth;

use App\Providers\RouteServiceProvider;
use Illuminate\Support\Facades\Auth;
use App\Models\User;
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

            // For testing purposes, let's create and login a test user
            $user = User::firstOrCreate(
                ['email' => 'test@example.com'],
                [
                    'name' => 'Test User',
                    'password' => bcrypt('password'),
                ]
            );

            // Manually authenticate the user
            Auth::login($user, $this->remember);
            
            session()->regenerate();
            
            logger('User authenticated successfully');
            
            return redirect()->intended(RouteServiceProvider::HOME);

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