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
        Schema::create('supplier', function (Blueprint $table) {
            $table->bigInteger('s_suppkey')->primary();
            $table->char('s_name', 30)->nullable();
            $table->string('s_address', 30)->nullable();
            $table->char('s_city', 20)->nullable();
            $table->char('s_nation', 20)->nullable();
            $table->char('s_region', 20)->nullable();
            $table->char('s_phone', 20)->nullable();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('supplier');
    }
};
