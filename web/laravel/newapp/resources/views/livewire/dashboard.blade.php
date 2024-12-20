<div>  {{-- Single root element for Livewire --}}
    <div class="py-12">
        <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
            <!-- Search -->
            <div class="mb-4">
                <input wire:model.live="searchTerm" type="text" 
                    class="rounded-md border-gray-300 shadow-sm w-full md:w-1/3" 
                    placeholder="Search customers...">
            </div>

            <!-- Summary Cards -->
            <div class="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <div class="text-gray-900 text-lg font-semibold">Customers</div>
                    <div class="text-3xl font-bold">{{ number_format($metrics['total_customers']) }}</div>
                </div>
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <div class="text-gray-900 text-lg font-semibold">Suppliers</div>
                    <div class="text-3xl font-bold">{{ number_format($metrics['total_suppliers']) }}</div>
                </div>
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <div class="text-gray-900 text-lg font-semibold">Parts</div>
                    <div class="text-3xl font-bold">{{ number_format($metrics['total_parts']) }}</div>
                </div>
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <div class="text-gray-900 text-lg font-semibold">Orders</div>
                    <div class="text-3xl font-bold">{{ number_format($metrics['total_orders']) }}</div>
                </div>
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <div class="text-gray-900 text-lg font-semibold">Total Revenue</div>
                    <div class="text-3xl font-bold">${{ number_format($metrics['total_revenue']) }}</div>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                <!-- Top Customers -->
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <h3 class="text-lg font-semibold mb-4">Top Customers by Revenue</h3>
                    <table class="min-w-full">
                        <thead>
                            <tr>
                                <th class="text-left">Customer</th>
                                <th class="text-right">Revenue</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($topCustomers as $customer)
                            <tr>
                                <td class="py-2">{{ $customer->c_name }}</td>
                                <td class="text-right">${{ number_format($customer->total_revenue) }}</td>
                            </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>

                <!-- Revenue by Region -->
                <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                    <h3 class="text-lg font-semibold mb-4">Revenue by Region</h3>
                    <table class="min-w-full">
                        <thead>
                            <tr>
                                <th class="text-left">Region</th>
                                <th class="text-right">Revenue</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($revenueByRegion as $region)
                            <tr>
                                <td class="py-2">{{ $region->c_region }}</td>
                                <td class="text-right">${{ number_format($region->total_revenue) }}</td>
                            </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Recent Orders -->
            <div class="bg-white overflow-hidden shadow-sm sm:rounded-lg p-6">
                <h3 class="text-lg font-semibold mb-4">Recent Orders</h3>
                <div class="overflow-x-auto">
                    <table class="min-w-full">
                        <thead>
                            <tr>
                                <th class="text-left">Order ID</th>
                                <th class="text-left">Customer</th>
                                <th class="text-left">Part</th>
                                <th class="text-right">Quantity</th>
                                <th class="text-right">Revenue</th>
                                <th class="text-left">Date</th>
                            </tr>
                        </thead>
                        <tbody>
                            @foreach($recentOrders as $order)
                            <tr>
                                <td class="py-2">{{ $order->lo_orderkey }}</td>
                                <td>{{ $order->customer?->c_name ?? 'N/A' }}</td>
                                <td>{{ $order->part?->p_name ?? 'N/A' }}</td>
                                <td class="text-right">{{ number_format($order->lo_quantity) }}</td>
                                <td class="text-right">${{ number_format($order->lo_revenue) }}</td>
                                <td>{{ $order->orderDate?->d_date ?? 'N/A' }}</td>
                            </tr>
                            @endforeach
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</div>