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
        Schema::create('lineorder', function (Blueprint $table) {
            $table->bigInteger('lo_orderkey');
            $table->bigInteger('lo_linenumber');
            $table->bigInteger('lo_custkey')->nullable()->index('lineorder_fk1');
            $table->bigInteger('lo_partkey')->nullable()->index('lineorder_fk2');
            $table->bigInteger('lo_suppkey')->nullable()->index('lineorder_fk3');
            $table->bigInteger('lo_orderdate')->nullable()->index('lineorder_fk4');
            $table->char('lo_orderpriority', 20)->nullable();
            $table->char('lo_shippriority', 1)->nullable();
            $table->bigInteger('lo_quantity')->nullable();
            $table->bigInteger('lo_extendedprice')->nullable();
            $table->bigInteger('lo_ordtotalprice')->nullable();
            $table->bigInteger('lo_discount')->nullable();
            $table->bigInteger('lo_revenue')->nullable();
            $table->bigInteger('lo_supplycost')->nullable();
            $table->bigInteger('lo_tax')->nullable();
            $table->bigInteger('lo_commitdate')->nullable()->index('lineorder_fk5');
            $table->char('lo_shipmode', 10)->nullable();

            $table->primary(['lo_orderkey', 'lo_linenumber']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('lineorder');
    }
};
