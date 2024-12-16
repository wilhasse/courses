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
        Schema::table('lineorder', function (Blueprint $table) {
            $table->foreign(['lo_custkey'], 'lineorder_ibfk_1')->references(['c_custkey'])->on('customer')->onUpdate('no action')->onDelete('no action');
            $table->foreign(['lo_partkey'], 'lineorder_ibfk_2')->references(['p_partkey'])->on('part')->onUpdate('no action')->onDelete('no action');
            $table->foreign(['lo_suppkey'], 'lineorder_ibfk_3')->references(['s_suppkey'])->on('supplier')->onUpdate('no action')->onDelete('no action');
            $table->foreign(['lo_orderdate'], 'lineorder_ibfk_4')->references(['d_datekey'])->on('date')->onUpdate('no action')->onDelete('no action');
            $table->foreign(['lo_commitdate'], 'lineorder_ibfk_5')->references(['d_datekey'])->on('date')->onUpdate('no action')->onDelete('no action');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('lineorder', function (Blueprint $table) {
            $table->dropForeign('lineorder_ibfk_1');
            $table->dropForeign('lineorder_ibfk_2');
            $table->dropForeign('lineorder_ibfk_3');
            $table->dropForeign('lineorder_ibfk_4');
            $table->dropForeign('lineorder_ibfk_5');
        });
    }
};
