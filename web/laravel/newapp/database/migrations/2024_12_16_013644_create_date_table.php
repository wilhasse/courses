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
        Schema::create('date', function (Blueprint $table) {
            $table->bigInteger('d_datekey')->primary();
            $table->char('d_date', 20)->nullable();
            $table->char('d_dayofweek', 10)->nullable();
            $table->char('d_month', 10)->nullable();
            $table->bigInteger('d_year')->nullable();
            $table->bigInteger('d_yearmonthnum')->nullable();
            $table->char('d_yearmonth', 10)->nullable();
            $table->bigInteger('d_daynuminmonth')->nullable();
            $table->bigInteger('d_daynuminyear')->nullable();
            $table->bigInteger('d_monthnuminyear')->nullable();
            $table->bigInteger('d_weeknuminyear')->nullable();
            $table->char('d_sellingseason', 20)->nullable();
            $table->bigInteger('d_lastdayinweekfl')->nullable();
            $table->bigInteger('d_lastdayinmonthfl')->nullable();
            $table->bigInteger('d_holidayfl')->nullable();
            $table->bigInteger('d_weekdayfl')->nullable();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('date');
    }
};
