<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Customer extends Model
{
    protected $table = 'customer';
    protected $primaryKey = 'c_custkey';
    public $timestamps = false;

    protected $fillable = [
        'c_name',
        'c_address',
        'c_city',
        'c_nation',
        'c_region',
        'c_phone',
        'c_mktsegment'
    ];

    public function lineOrders()
    {
        return $this->hasMany(LineOrder::class, 'lo_custkey', 'c_custkey');
    }
}