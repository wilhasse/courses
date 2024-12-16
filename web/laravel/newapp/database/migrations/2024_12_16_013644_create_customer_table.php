<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('customer', function (Blueprint $table) {
            $table->bigInteger('c_custkey')->primary();
            $table->string('c_name', 30)->nullable();
            $table->string('c_address', 30)->nullable();
            $table->char('c_city', 20)->nullable();
            $table->char('c_nation', 20)->nullable();
            $table->char('c_region', 20)->nullable();
            $table->char('c_phone', 20)->nullable();
            $table->char('c_mktsegment', 20)->nullable();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('customer');
    }
};
