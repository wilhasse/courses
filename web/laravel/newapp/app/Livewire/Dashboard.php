<?php

namespace App\Livewire;

use App\Models\Customer;
use App\Models\LineOrder;
use App\Models\Part;
use App\Models\Supplier;
use Illuminate\Support\Facades\DB;
use Livewire\Component;

class Dashboard extends Component
{
    public $selectedPeriod = 'all';
    public $searchTerm = '';

    public function render()
    {
        // Summary metrics
        $metrics = [
            'total_customers' => Customer::count(),
            'total_suppliers' => Supplier::count(),
            'total_parts' => Part::count(),
            'total_orders' => LineOrder::count(),
            'total_revenue' => LineOrder::sum('lo_revenue'),
        ];

        // Top customers query
        $topCustomersQuery = LineOrder::select('customer.c_name', DB::raw('SUM(lo_revenue) as total_revenue'))
            ->join('customer', 'lo_custkey', '=', 'c_custkey')
            ->groupBy('customer.c_name');

        if ($this->searchTerm) {
            $topCustomersQuery->where('customer.c_name', 'like', '%' . $this->searchTerm . '%');
        }

        $topCustomers = $topCustomersQuery
            ->orderByDesc('total_revenue')
            ->limit(5)
            ->get();

        // Recent orders with relationships
        $recentOrders = LineOrder::with(['customer', 'part', 'orderDate'])
            ->orderBy('lo_orderkey', 'desc')
            ->limit(10)
            ->get();

        return view('livewire.dashboard', [
            'metrics' => $metrics,
            'topCustomers' => $topCustomers,
            'recentOrders' => $recentOrders,
            'revenueByRegion' => $this->getRevenueByRegion(),
        ])->layout('layouts.app');
    }

    private function getRevenueByRegion()
    {
        return LineOrder::select('customer.c_region', DB::raw('SUM(lo_revenue) as total_revenue'))
            ->join('customer', 'lo_custkey', '=', 'c_custkey')
            ->groupBy('customer.c_region')
            ->orderByDesc('total_revenue')
            ->get();
    }
}