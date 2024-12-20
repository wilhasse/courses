<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Supplier extends Model
{
    protected $table = 'supplier';
    protected $primaryKey = 's_suppkey';
    public $timestamps = false;

    protected $fillable = [
        's_name',
        's_address',
        's_city',
        's_nation',
        's_region',
        's_phone'
    ];

    public function lineOrders()
    {
        return $this->hasMany(LineOrder::class, 'lo_suppkey', 's_suppkey');
    }
}