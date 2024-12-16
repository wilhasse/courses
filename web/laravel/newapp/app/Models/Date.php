<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Date extends Model
{
    protected $table = 'date';
    protected $primaryKey = 'd_datekey';
    public $timestamps = false;

    protected $fillable = [
        'd_date',
        'd_dayofweek',
        'd_month',
        'd_year',
        'd_yearmonthnum',
        'd_yearmonth',
        'd_daynuminmonth',
        'd_daynuminyear',
        'd_monthnuminyear',
        'd_weeknuminyear',
        'd_sellingseason',
        'd_lastdayinweekfl',
        'd_lastdayinmonthfl',
        'd_holidayfl',
        'd_weekdayfl'
    ];

    public function orderDates()
    {
        return $this->hasMany(LineOrder::class, 'lo_orderdate', 'd_datekey');
    }

    public function commitDates()
    {
        return $this->hasMany(LineOrder::class, 'lo_commitdate', 'd_datekey');
    }
}