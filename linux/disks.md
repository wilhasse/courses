# Tool

fio  
Author: Jens Axboe (Linux kernel maintainer)
https://fio.readthedocs.io/en/latest/fio_doc.html

# Testing Mix random read/write

```bash
fio --name=mysql-innodb-test \
    --filename=iotest \
    --size=10G \
    --ioengine=libaio \
    --direct=1 \
    --gtod_reduce=1 \
    --randrepeat=1 \
    --iodepth=64 \
    --numjobs=4 \
    --runtime=300 \
    --time_based \
    --group_reporting \
    --readwrite=randrw \
    --rwmixread=70 \
    --bs=16K \
    --rate_iops=1000 \
    --ramp_time=10
```

Example

```
   read: IOPS=5444, BW=21.3MiB/s (22.3MB/s)(514MiB/24179msec)
   bw (  KiB/s): min= 3624, max=51928, per=100.00%, avg=21887.17, stdev=13914.88, samples=48
   iops        : min=  906, max=12982, avg=5471.79, stdev=3478.72, samples=48
```

## Testing Latency

```bash
fio --name=latency-test \
    --filename=iotest \
    --direct=1 \
    --rw=randread \
    --bs=4k \
    --ioengine=libaio \
    --iodepth=1 \
    --size=10G \
    --runtime=60 \
    --numjobs=1 \
    --time_based \
    --group_reporting
```

Example:

```
    clat percentiles (usec):
     |  1.00th=[   58],  5.00th=[   58], 10.00th=[   59], 20.00th=[   59],
     | 30.00th=[   59], 40.00th=[   59], 50.00th=[   59], 60.00th=[   60],
     | 70.00th=[   60], 80.00th=[   60], 90.00th=[   61], 95.00th=[   63],
     | 99.00th=[   71], 99.50th=[   80], 99.90th=[  120], 99.95th=[  139],
     | 99.99th=[  210]
```
