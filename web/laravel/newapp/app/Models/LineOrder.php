<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class LineOrder extends Model
{
    protected $table = 'lineorder';
    protected $primaryKey = ['lo_orderkey', 'lo_linenumber'];
    public $incrementing = false;
    public $timestamps = false;

    protected $fillable = [
        'lo_custkey',
        'lo_partkey',
        'lo_suppkey',
        'lo_orderdate',
        'lo_orderpriority',
        'lo_shippriority',
        'lo_quantity',
        'lo_extendedprice',
        'lo_ordtotalprice',
        'lo_discount',
        'lo_revenue',
        'lo_supplycost',
        'lo_tax',
        'lo_commitdate',
        'lo_shipmode'
    ];

    public function customer()
    {
        return $this->belongsTo(Customer::class, 'lo_custkey', 'c_custkey');
    }

    public function part()
    {
        return $this->belongsTo(Part::class, 'lo_partkey', 'p_partkey');
    }

    public function supplier()
    {
        return $this->belongsTo(Supplier::class, 'lo_suppkey', 's_suppkey');
    }

    public function orderDate()
    {
        return $this->belongsTo(Date::class, 'lo_orderdate', 'd_datekey');
    }

    public function commitDate()
    {
        return $this->belongsTo(Date::class, 'lo_commitdate', 'd_datekey');
    }
}