<!DOCTYPE html>
<html lang="{{ str_replace('_', '-', app()->getLocale()) }}">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ $title ?? config('app.name') }}</title>
    
    @vite(['resources/css/app.css', 'resources/js/app.js'])
    @fluxStyles
    @livewireStyles
</head>
<body class="min-h-screen bg-gray-100">
    <!-- Navigation -->
    <nav class="bg-white shadow">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16">
                <div class="flex">
                    <!-- Logo -->
                    <div class="flex-shrink-0 flex items-center">
                        <span class="text-xl font-bold">TI CSLOG</span>
                    </div>
                </div>

                <!-- Right side navigation -->
                <div class="flex items-center">
                    <div class="ml-3 relative">
                        @auth
                            <div class="flex items-center space-x-4">
                                <span class="text-gray-700">{{ Auth::user()->name }}</span>
                                <form method="POST" action="{{ route('logout') }}">
                                    @csrf
                                    <button type="submit" 
                                        class="text-gray-600 hover:text-gray-900">
                                        Logout
                                    </button>
                                </form>
                            </div>
                        @endauth
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Page Content -->
    <main>
        {{ $slot }}
    </main>
    
    @fluxScripts
    @livewireScripts
</body>
</html> 