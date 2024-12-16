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
        Schema::create('part', function (Blueprint $table) {
            $table->bigInteger('p_partkey')->primary();
            $table->string('p_name', 30)->nullable();
            $table->char('p_mfgr', 10)->nullable();
            $table->char('p_category', 10)->nullable();
            $table->char('p_brand1', 10)->nullable();
            $table->string('p_color', 20)->nullable();
            $table->string('p_type', 30)->nullable();
            $table->bigInteger('p_size')->nullable();
            $table->char('p_container', 10)->nullable();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('part');
    }
};
