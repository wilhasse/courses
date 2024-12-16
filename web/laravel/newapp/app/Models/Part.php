<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Part extends Model
{
    protected $table = 'part';
    protected $primaryKey = 'p_partkey';
    public $timestamps = false;

    protected $fillable = [
        'p_name',
        'p_mfgr',
        'p_category',
        'p_brand1',
        'p_color',
        'p_type',
        'p_size',
        'p_container'
    ];

    public function lineOrders()
    {
        return $this->hasMany(LineOrder::class, 'lo_partkey', 'p_partkey');
    }
}